# Strava for Figure Skating — Coach Dashboard MVP

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the one-shot analysis tool into a Strava-like platform with session history, progress tracking, coach-skater relationships, and automatic diagnostics.

**Architecture:** Backend extends the existing FastAPI app with new Postgres tables (sessions, session_metrics, relationships), new API routes, and services (PR tracker, diagnostics engine). Frontend is redesigned with new routes (feed, upload, progress, dashboard) using shared components. Pipeline integration saves analysis results to Postgres after successful processing.

**Tech Stack:** FastAPI + SQLAlchemy 2.0 async + Alembic (backend), Next.js 16 + React Query + Recharts + shadcn/ui (frontend), PostgreSQL (data), R2/S3 (storage).

**Depends on:** Auth spec (data/specs/2026-04-11-saas-auth-db-profiles-design.md) — Postgres, User model, JWT auth must be implemented first.

**Design spec:** data/specs/2026-04-11-strava-fs-design.md

---

## File Structure

### Backend — New files

| File | Responsibility |
|------|---------------|
| `src/backend/metrics_registry.py` | METRIC_REGISTRY dict + MetricDef dataclass |
| `src/backend/models/session.py` | Session, SessionMetric ORM models |
| `src/backend/models/relationship.py` | Relationship ORM model |
| `src/backend/crud/session.py` | Session CRUD operations |
| `src/backend/crud/session_metric.py` | SessionMetric CRUD operations |
| `src/backend/crud/relationship.py` | Relationship CRUD operations |
| `src/backend/services/pr_tracker.py` | PR detection logic |
| `src/backend/services/diagnostics.py` | Diagnostic rules engine |
| `src/backend/services/session_saver.py` | Save pipeline results to Postgres |
| `src/backend/routes/sessions.py` | Session API routes (CRUD) |
| `src/backend/routes/metrics.py` | Metrics API routes (trend, PRs, diagnostics, registry) |
| `src/backend/routes/relationships.py` | Relationship API routes |
| `src/backend/routes/uploads.py` | Chunked upload endpoints |
| `alembic/versions/xxxx_add_sessions_and_relationships.py` | Migration |

### Backend — Modified files

| File | Change |
|------|--------|
| `src/backend/models/__init__.py` | Export new models |
| `src/backend/main.py` | Register new route modules |
| `src/backend/schemas.py` | Add session, metric, relationship schemas |
| `src/web_helpers.py` | Return AnalysisReport from pipeline |
| `src/worker.py` | Call session_saver after successful processing |

### Frontend — New files

| File | Responsibility |
|------|---------------|
| `src/frontend/src/lib/api/sessions.ts` | Session API hooks (React Query) |
| `src/frontend/src/lib/api/metrics.ts` | Metrics API hooks |
| `src/frontend/src/lib/api/relationships.ts` | Relationship API hooks |
| `src/frontend/src/lib/api/uploads.ts` | Chunked upload client |
| `src/frontend/src/lib/metrics-context.ts` | Metric registry client-side |
| `src/frontend/src/components/layout/bottom-tabs.tsx` | Mobile bottom tab bar |
| `src/frontend/src/components/layout/sidebar.tsx` | Desktop sidebar nav |
| `src/frontend/src/components/layout/app-shell.tsx` | Responsive shell (tabs vs sidebar) |
| `src/frontend/src/components/session/session-card.tsx` | Activity feed card |
| `src/frontend/src/components/session/metric-row.tsx` | Single metric display with color |
| `src/frontend/src/components/session/metric-badge.tsx` | PR badge component |
| `src/frontend/src/components/upload/element-picker.tsx` | Element type grid selector |
| `src/frontend/src/components/upload/camera-recorder.tsx` | In-browser video recording |
| `src/frontend/src/components/upload/chunked-uploader.tsx` | Chunked upload with progress |
| `src/frontend/src/components/progress/trend-chart.tsx` | Recharts line chart |
| `src/frontend/src/components/progress/period-selector.tsx` | 7d/30d/90d/all selector |
| `src/frontend/src/components/coach/student-card.tsx` | Coach dashboard student card |
| `src/frontend/src/components/coach/diagnostics-list.tsx` | Diagnostics findings list |
| `src/frontend/src/app/(app)/feed/page.tsx` | Activity feed page |
| `src/frontend/src/app/(app)/upload/page.tsx` | Upload page |
| `src/frontend/src/app/(app)/sessions/[id]/page.tsx` | Session detail page |
| `src/frontend/src/app/(app)/progress/page.tsx` | Progress charts page |
| `src/frontend/src/app/(app)/connections/page.tsx` | Coach-skater connections |
| `src/frontend/src/app/(app)/dashboard/page.tsx` | Coach roster dashboard |
| `src/frontend/src/app/(app)/students/[id]/page.tsx` | Student profile (progress + diagnostics) |
| `src/frontend/src/app/(app)/layout.tsx` | App layout with shell |

### Frontend — Modified files

| File | Change |
|------|--------|
| `src/frontend/src/types/index.ts` | Add session, metric, relationship types |
| `src/frontend/src/app/layout.tsx` | Redirect logic (logged out → login) |
| `src/frontend/package.json` | Add recharts dependency |
| `src/frontend/src/lib/api-client.ts` | Add `apiPost`, `apiPatch`, `apiDelete` helpers |

---

## Task 1: Metric Registry

**Files:**
- Create: `src/backend/metrics_registry.py`
- Test: `tests/backend/test_metrics_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/backend/test_metrics_registry.py
"""Tests for the metric registry."""

import pytest

from src.backend.metrics_registry import METRIC_REGISTRY, MetricDef


def test_registry_has_all_known_metrics():
    """Registry must contain every metric produced by BiomechanicsAnalyzer."""
    expected = {
        "airtime", "max_height", "relative_jump_height",
        "landing_knee_angle", "landing_knee_stability", "landing_trunk_recovery",
        "arm_position_score", "rotation_speed",
        "knee_angle", "trunk_lean", "edge_change_smoothness",
        "symmetry",
    }
    assert set(METRIC_REGISTRY.keys()) == expected


def test_metric_def_fields():
    """Each MetricDef must have all required fields."""
    for name, mdef in METRIC_REGISTRY.items():
        assert mdef.name == name
        assert isinstance(mdef.label_ru, str) and len(mdef.label_ru) > 0
        assert mdef.unit in {"s", "deg", "score", "norm", "ratio", "deg/s"}
        assert mdef.format.startswith(".")
        assert mdef.direction in {"higher", "lower"}
        assert len(mdef.element_types) > 0
        assert isinstance(mdef.ideal_range, tuple) and len(mdef.ideal_range) == 2
        assert mdef.ideal_range[0] < mdef.ideal_range[1]


def test_jump_metrics_only_on_jump_elements():
    """Jump-specific metrics must not appear on three_turn."""
    jump_only = {"airtime", "max_height", "rotation_speed", "landing_knee_angle"}
    three_turn_metrics = {
        name for name, mdef in METRIC_REGISTRY.items()
        if "three_turn" in mdef.element_types
    }
    assert not (jump_only & three_turn_metrics) or len(jump_only & three_turn_metrics) == 0


def test_symmetry_on_all_elements():
    """Symmetry metric must be available for all element types."""
    all_elements = {
        "waltz_jump", "toe_loop", "flip", "salchow", "loop", "lutz", "axel", "three_turn"
    }
    assert set(METRIC_REGISTRY["symmetry"].element_types) == all_elements
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/backend/test_metrics_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.backend.metrics_registry'`

- [ ] **Step 3: Write the metric registry**

```python
# src/backend/metrics_registry.py
"""Central metric definitions — single source of truth for backend and frontend."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MetricDef:
    """Definition of a single biomechanical metric."""

    name: str
    label_ru: str
    unit: str  # "s", "deg", "score", "norm", "ratio", "deg/s"
    format: str  # Python format spec, e.g. ".2f"
    direction: str  # "higher" or "lower"
    element_types: tuple[str, ...]  # which elements produce this metric
    ideal_range: tuple[float, float]


METRIC_REGISTRY: dict[str, MetricDef] = {
    "airtime": MetricDef(
        name="airtime", label_ru="Время полёта", unit="s", format=".2f",
        direction="higher",
        element_types=("waltz_jump", "toe_loop", "flip", "salchow", "loop", "lutz", "axel"),
        ideal_range=(0.3, 0.7),
    ),
    "max_height": MetricDef(
        name="max_height", label_ru="Высота прыжка", unit="norm", format=".3f",
        direction="higher",
        element_types=("waltz_jump", "toe_loop", "flip", "salchow", "loop", "lutz", "axel"),
        ideal_range=(0.2, 0.5),
    ),
    "relative_jump_height": MetricDef(
        name="relative_jump_height", label_ru="Относительная высота", unit="ratio", format=".2f",
        direction="higher",
        element_types=("waltz_jump", "toe_loop", "flip", "salchow", "loop", "lutz", "axel"),
        ideal_range=(0.3, 1.5),
    ),
    "landing_knee_angle": MetricDef(
        name="landing_knee_angle", label_ru="Угол колена при приземлении", unit="deg", format=".0f",
        direction="higher",
        element_types=("waltz_jump", "toe_loop", "flip", "salchow", "loop", "lutz", "axel"),
        ideal_range=(90, 130),
    ),
    "landing_knee_stability": MetricDef(
        name="landing_knee_stability", label_ru="Стабильность приземления", unit="score", format=".2f",
        direction="higher",
        element_types=("waltz_jump", "toe_loop", "flip", "salchow", "loop", "lutz", "axel"),
        ideal_range=(0.5, 1.0),
    ),
    "landing_trunk_recovery": MetricDef(
        name="landing_trunk_recovery", label_ru="Восстановление корпуса", unit="score", format=".2f",
        direction="higher",
        element_types=("waltz_jump", "toe_loop", "flip", "salchow", "loop", "lutz", "axel"),
        ideal_range=(0.5, 1.0),
    ),
    "arm_position_score": MetricDef(
        name="arm_position_score", label_ru="Контроль рук", unit="score", format=".2f",
        direction="higher",
        element_types=("waltz_jump", "toe_loop", "flip", "salchow", "loop", "lutz", "axel"),
        ideal_range=(0.6, 1.0),
    ),
    "rotation_speed": MetricDef(
        name="rotation_speed", label_ru="Скорость вращения", unit="deg/s", format=".0f",
        direction="higher",
        element_types=("waltz_jump", "toe_loop", "flip", "salchow", "loop", "lutz", "axel"),
        ideal_range=(300, 550),
    ),
    "knee_angle": MetricDef(
        name="knee_angle", label_ru="Угол колена", unit="deg", format=".0f",
        direction="higher",
        element_types=("three_turn",),
        ideal_range=(100, 140),
    ),
    "trunk_lean": MetricDef(
        name="trunk_lean", label_ru="Наклон корпуса", unit="deg", format=".1f",
        direction="higher",
        element_types=("three_turn",),
        ideal_range=(-15, 20),
    ),
    "edge_change_smoothness": MetricDef(
        name="edge_change_smoothness", label_ru="Плавность смены ребра", unit="score", format=".2f",
        direction="higher",
        element_types=("three_turn",),
        ideal_range=(0.1, 0.5),
    ),
    "symmetry": MetricDef(
        name="symmetry", label_ru="Симметрия", unit="score", format=".2f",
        direction="higher",
        element_types=("waltz_jump", "toe_loop", "flip", "salchow", "loop", "lutz", "axel", "three_turn"),
        ideal_range=(0.6, 1.0),
    ),
}


def get_metrics_for_element(element_type: str) -> dict[str, MetricDef]:
    """Return metrics applicable to a given element type."""
    return {name: mdef for name, mdef in METRIC_REGISTRY.items() if element_type in mdef.element_types}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/backend/test_metrics_registry.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/backend/metrics_registry.py tests/backend/test_metrics_registry.py
git commit -m "feat(backend): add metric registry with all 12 biomechanical metrics"
```

---

## Task 2: ORM Models + Migration

**Files:**
- Create: `src/backend/models/session.py`
- Create: `src/backend/models/relationship.py`
- Modify: `src/backend/models/__init__.py`
- Create: `alembic/versions/xxxx_add_sessions_and_relationships.py`

- [ ] **Step 1: Write ORM models**

```python
# src/backend/models/session.py
"""Session and SessionMetric ORM models."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, String, func
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.backend.models.base import Base, TimestampMixin


class Session(TimestampMixin, Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), index=True,
    )
    element_type: Mapped[str] = mapped_column(String(50), index=True)
    video_url: Mapped[str | None] = mapped_column(String(500))
    processed_video_url: Mapped[str | None] = mapped_column(String(500))
    poses_url: Mapped[str | None] = mapped_column(String(500))
    csv_url: Mapped[str | None] = mapped_column(String(500))
    status: Mapped[str] = mapped_column(String(20), default="uploading")
    error_message: Mapped[str | None]

    # Analysis results (denormalized for fast reads)
    phases: Mapped[dict | None] = mapped_column(JSON)
    recommendations: Mapped[list | None] = mapped_column(JSON)
    overall_score: Mapped[float | None] = mapped_column(Float)

    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Relationships
    metrics: Mapped[list[SessionMetric]] = relationship(back_populates="session", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_sessions_user_element_created", "user_id", "element_type", "created_at"),
    )


class SessionMetric(TimestampMixin, Base):
    __tablename__ = "session_metrics"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sessions.id", ondelete="CASCADE"), index=True,
    )
    metric_name: Mapped[str] = mapped_column(String(100), index=True)
    metric_value: Mapped[float] = mapped_column(Float)

    # PR tracking
    is_pr: Mapped[bool] = mapped_column(Boolean, default=False)
    prev_best: Mapped[float | None] = mapped_column(Float)

    # Reference comparison
    reference_value: Mapped[float | None] = mapped_column(Float)
    is_in_range: Mapped[bool | None] = mapped_column(Boolean)

    # Relationships
    session: Mapped[Session] = relationship(back_populates="metrics")

    __table_args__ = (
        Index("uq_session_metric", "session_id", "metric_name", unique=True),
    )
```

```python
# src/backend/models/relationship.py
"""Relationship ORM model for coach-skater connections."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, func
from sqlalchemy.orm import Mapped, mapped_column

from src.backend.models.base import Base


class Relationship(Base):
    __tablename__ = "relationships"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    coach_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"),
    )
    skater_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"),
    )
    status: Mapped[str] = mapped_column(String(20), default="invited")
    initiated_by: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"),
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        # No duplicate active relationships between same coach-skater pair
        Index(
            "uq_coach_skater_active",
            "coach_id", "skater_id",
            unique=True,
            postgresql_where="status != 'ended'",
        ),
    )
```

- [ ] **Step 2: Update models __init__.py**

```python
# src/backend/models/__init__.py
"""SQLAlchemy ORM models."""

from src.backend.models.base import Base
from src.backend.models.refresh_token import RefreshToken
from src.backend.models.relationship import Relationship
from src.backend.models.session import Session, SessionMetric
from src.backend.models.user import User

__all__ = ["Base", "RefreshToken", "Relationship", "Session", "SessionMetric", "User"]
```

- [ ] **Step 3: Generate and review Alembic migration**

Run: `uv run alembic revision --autogenerate -m "add_sessions_and_relationships"`

Review the generated migration file. Verify it creates:
- `sessions` table with all columns and indexes
- `session_metrics` table with unique constraint
- `relationships` table with partial unique index

Then apply:
Run: `uv run alembic upgrade head`

- [ ] **Step 4: Commit**

```bash
git add src/backend/models/session.py src/backend/models/relationship.py src/backend/models/__init__.py alembic/versions/
git commit -m "feat(backend): add Session, SessionMetric, Relationship ORM models + migration"
```

---

## Task 3: CRUD Operations

**Files:**
- Create: `src/backend/crud/session.py`
- Create: `src/backend/crud/session_metric.py`
- Create: `src/backend/crud/relationship.py`
- Test: `tests/backend/test_crud_session.py`

- [ ] **Step 1: Write session CRUD**

```python
# src/backend/crud/session.py
"""Session CRUD operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import desc, select
from sqlalchemy.orm import selectinload

from src.backend.models.session import Session

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def create(db: AsyncSession, *, user_id: str, element_type: str, **kwargs) -> Session:
    session = Session(user_id=user_id, element_type=element_type, **kwargs)
    db.add(session)
    await db.flush()
    await db.refresh(session)
    return session


async def get_by_id(db: AsyncSession, session_id: str) -> Session | None:
    result = await db.execute(
        select(Session).options(selectinload(Session.metrics)).where(Session.id == session_id)
    )
    return result.scalar_one_or_none()


async def list_by_user(
    db: AsyncSession,
    user_id: str,
    *,
    element_type: str | None = None,
    limit: int = 20,
    offset: int = 0,
    sort: str = "created_at",
) -> list[Session]:
    query = select(Session).where(Session.user_id == user_id)
    if element_type:
        query = query.where(Session.element_type == element_type)
    if sort == "overall_score":
        query = query.order_by(desc(Session.overall_score))
    else:
        query = query.order_by(desc(Session.created_at))
    query = query.offset(offset).limit(limit)
    result = await db.execute(query)
    return list(result.scalars().all())


async def update(db: AsyncSession, session: Session, **kwargs) -> Session:
    for key, value in kwargs.items():
        if value is not None:
            setattr(session, key, value)
    db.add(session)
    await db.flush()
    await db.refresh(session)
    return session


async def soft_delete(db: AsyncSession, session: Session) -> None:
    session.status = "deleted"
    db.add(session)
    await db.flush()
```

- [ ] **Step 2: Write session metric CRUD**

```python
# src/backend/crud/session_metric.py
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
```

- [ ] **Step 3: Write relationship CRUD**

```python
# src/backend/crud/relationship.py
"""Relationship CRUD operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import and_, select

from src.backend.models.relationship import Relationship

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def create(
    db: AsyncSession, *, coach_id: str, skater_id: str, initiated_by: str,
) -> Relationship:
    rel = Relationship(coach_id=coach_id, skater_id=skater_id, initiated_by=initiated_by)
    db.add(rel)
    await db.flush()
    await db.refresh(rel)
    return rel


async def get_by_id(db: AsyncSession, rel_id: str) -> Relationship | None:
    result = await db.execute(select(Relationship).where(Relationship.id == rel_id))
    return result.scalar_one_or_none()


async def get_active(
    db: AsyncSession, coach_id: str, skater_id: str,
) -> Relationship | None:
    result = await db.execute(
        select(Relationship).where(
            Relationship.coach_id == coach_id,
            Relationship.skater_id == skater_id,
            Relationship.status != "ended",
        )
    )
    return result.scalar_one_or_none()


async def list_for_user(db: AsyncSession, user_id: str) -> list[Relationship]:
    """List all relationships where user is coach or skater."""
    result = await db.execute(
        select(Relationship).where(
            (Relationship.coach_id == user_id) | (Relationship.skater_id == user_id),
        ).order_by(Relationship.created_at.desc())
    )
    return list(result.scalars().all())


async def list_pending_for_skater(db: AsyncSession, skater_id: str) -> list[Relationship]:
    result = await db.execute(
        select(Relationship).where(
            Relationship.skater_id == skater_id,
            Relationship.status == "invited",
        ).order_by(Relationship.created_at.desc())
    )
    return list(result.scalars().all())


async def list_active_students(db: AsyncSession, coach_id: str) -> list[Relationship]:
    result = await db.execute(
        select(Relationship).where(
            Relationship.coach_id == coach_id,
            Relationship.status == "active",
        ).order_by(Relationship.created_at.desc())
    )
    return list(result.scalars().all())


async def list_active_coaches(db: AsyncSession, skater_id: str) -> list[Relationship]:
    result = await db.execute(
        select(Relationship).where(
            Relationship.skater_id == skater_id,
            Relationship.status == "active",
        ).order_by(Relationship.created_at.desc())
    )
    return list(result.scalars().all())


async def is_coach_for_student(db: AsyncSession, coach_id: str, skater_id: str) -> bool:
    result = await db.execute(
        select(Relationship).where(
            Relationship.coach_id == coach_id,
            Relationship.skater_id == skater_id,
            Relationship.status == "active",
        )
    )
    return result.scalar_one_or_none() is not None
```

- [ ] **Step 4: Commit**

```bash
git add src/backend/crud/session.py src/backend/crud/session_metric.py src/backend/crud/relationship.py
git commit -m "feat(backend): add CRUD for sessions, session_metrics, relationships"
```

---

## Task 4: PR Tracker Service

**Files:**
- Create: `src/backend/services/pr_tracker.py`
- Test: `tests/backend/test_pr_tracker.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/backend/test_pr_tracker.py
"""Tests for PR tracker service."""

import pytest

from src.backend.services.pr_tracker import check_pr


@pytest.mark.parametrize(
    "direction,current_best,new_value,expected_is_pr",
    [
        ("higher", 0.38, 0.42, True),   # new value beats best
        ("higher", 0.42, 0.38, False),   # lower than best
        ("higher", None, 0.42, True),    # first ever
        ("lower", 0.10, 0.05, True),    # new value is lower (better)
        ("lower", 0.05, 0.10, False),   # higher than best
        ("lower", None, 0.05, True),    # first ever
    ],
)
def test_check_pr(direction, current_best, new_value, expected_is_pr):
    is_pr, prev_best = check_pr(direction, current_best, new_value)
    assert is_pr == expected_is_pr
    if expected_is_pr:
        assert prev_best == current_best
    else:
        assert prev_best is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/backend/test_pr_tracker.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement PR tracker**

```python
# src/backend/services/pr_tracker.py
"""Personal record detection logic."""


def check_pr(
    direction: str, current_best: float | None, new_value: float,
) -> tuple[bool, float | None]:
    """Check if new_value is a personal record.

    Args:
        direction: "higher" (bigger is better) or "lower" (smaller is better).
        current_best: Previous best value, or None if no history.
        new_value: The new metric value to check.

    Returns:
        (is_pr, prev_best) where prev_best is the old best if it's a PR.
    """
    if current_best is None:
        return True, None

    if direction == "higher":
        is_pr = new_value > current_best
    else:
        is_pr = new_value < current_best

    return is_pr, current_best if is_pr else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/backend/test_pr_tracker.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/backend/services/pr_tracker.py tests/backend/test_pr_tracker.py
git commit -m "feat(backend): add PR tracker service"
```

---

## Task 5: Diagnostics Engine

**Files:**
- Create: `src/backend/services/diagnostics.py`
- Test: `tests/backend/test_diagnostics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/backend/test_diagnostics.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/backend/test_diagnostics.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement diagnostics engine**

```python
# src/backend/services/diagnostics.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/backend/test_diagnostics.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/backend/services/diagnostics.py tests/backend/test_diagnostics.py
git commit -m "feat(backend): add diagnostics engine with 5 rule checks"
```

---

## Task 6: Pydantic Schemas

**Files:**
- Modify: `src/backend/schemas.py`

- [ ] **Step 1: Add session, metric, relationship schemas**

Append to `src/backend/schemas.py`:

```python
# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    element_type: str = Field(..., min_length=1, max_length=50)


class PatchSessionRequest(BaseModel):
    element_type: str | None = Field(default=None, max_length=50)


class SessionMetricResponse(BaseModel):
    id: str
    metric_name: str
    metric_value: float
    is_pr: bool
    prev_best: float | None
    reference_value: float | None
    is_in_range: bool | None

    model_config = {"from_attributes": True}


class SessionResponse(BaseModel):
    id: str
    user_id: str
    element_type: str
    video_url: str | None
    processed_video_url: str | None
    poses_url: str | None
    csv_url: str | None
    status: str
    error_message: str | None
    phases: dict | None
    recommendations: list[str] | None
    overall_score: float | None
    created_at: str
    processed_at: str | None
    metrics: list[SessionMetricResponse] = []

    model_config = {"from_attributes": True}

    @field_validator("created_at", "processed_at", mode="before")
    @classmethod
    def validate_datetime(cls, v: Any) -> str:
        if isinstance(v, datetime):
            return v.isoformat()
        return str(v)


class SessionListResponse(BaseModel):
    sessions: list[SessionResponse]
    total: int


# ---------------------------------------------------------------------------
# Metrics & Progress
# ---------------------------------------------------------------------------

class TrendDataPoint(BaseModel):
    date: str
    value: float
    session_id: str
    is_pr: bool


class TrendResponse(BaseModel):
    metric_name: str
    element_type: str
    data_points: list[TrendDataPoint]
    trend: str  # improving | stable | declining
    current_pr: float | None
    reference_range: dict[str, float] | None


class DiagnosticsFinding(BaseModel):
    severity: str
    element: str
    metric: str
    message: str
    detail: str


class DiagnosticsResponse(BaseModel):
    user_id: str
    findings: list[DiagnosticsFinding]


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------

class InviteRequest(BaseModel):
    skater_email: str


class RelationshipResponse(BaseModel):
    id: str
    coach_id: str
    skater_id: str
    status: str
    initiated_by: str | None
    created_at: str
    ended_at: str | None
    coach_name: str | None = None
    skater_name: str | None = None

    model_config = {"from_attributes": True}

    @field_validator("created_at", "ended_at", mode="before")
    @classmethod
    def validate_datetime(cls, v: Any) -> str:
        if isinstance(v, datetime):
            return v.isoformat()
        return str(v)


class RelationshipListResponse(BaseModel):
    relationships: list[RelationshipResponse]
```

- [ ] **Step 2: Commit**

```bash
git add src/backend/schemas.py
git commit -m "feat(backend): add Pydantic schemas for sessions, metrics, relationships"
```

---

## Task 7: Session API Routes

**Files:**
- Create: `src/backend/routes/sessions.py`
- Modify: `src/backend/main.py`

- [ ] **Step 1: Write session routes**

```python
# src/backend/routes/sessions.py
"""Session CRUD API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from src.backend.auth.deps import CurrentUser, DbDep
from src.backend.crud.relationship import is_coach_for_student
from src.backend.crud.session import create, get_by_id, list_by_user, soft_delete, update
from src.backend.schemas import (
    CreateSessionRequest,
    PatchSessionRequest,
    SessionListResponse,
    SessionResponse,
)

router = APIRouter(tags=["sessions"])


def _session_to_response(session) -> SessionResponse:
    """Convert ORM Session to response schema."""
    return SessionResponse.model_validate(session)


@router.post("/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(body: CreateSessionRequest, user: CurrentUser, db: DbDep):
    session = await create(db, user_id=user.id, element_type=body.element_type)
    return _session_to_response(session)


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    user: CurrentUser,
    db: DbDep,
    user_id: str | None = None,
    element_type: str | None = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    sort: str = Query("created_at", pattern="^(created_at|overall_score)$"),
):
    # Coaches can view their students' sessions
    target_user_id = user_id if user_id else user.id
    if user_id and user_id != user.id:
        if not await is_coach_for_student(db, coach_id=user.id, skater_id=user_id):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not a coach for this user")

    sessions = await list_by_user(
        db, user_id=target_user_id, element_type=element_type, limit=limit, offset=offset, sort=sort,
    )
    return SessionListResponse(sessions=[_session_to_response(s) for s in sessions], total=len(sessions))


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str, user: CurrentUser, db: DbDep):
    session = await get_by_id(db, session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    if session.user_id != user.id:
        if not await is_coach_for_student(db, coach_id=user.id, skater_id=session.user_id):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    return _session_to_response(session)


@router.patch("/sessions/{session_id}", response_model=SessionResponse)
async def patch_session(
    session_id: str, body: PatchSessionRequest, user: CurrentUser, db: DbDep,
):
    session = await get_by_id(db, session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    if session.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    session = await update(db, session, **body.model_dump(exclude_unset=True))
    return _session_to_response(session)


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str, user: CurrentUser, db: DbDep):
    session = await get_by_id(db, session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    if session.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    await soft_delete(db, session)
```

- [ ] **Step 2: Register routes in main.py**

Add to `src/backend/main.py` imports:
```python
from src.backend.routes import auth, detect, metrics, misc, models, process, relationships, sessions, users
```

Add to `api_v1`:
```python
api_v1.include_router(sessions.router)
api_v1.include_router(metrics.router)
api_v1.include_router(relationships.router)
```

- [ ] **Step 3: Commit**

```bash
git add src/backend/routes/sessions.py src/backend/main.py
git commit -m "feat(backend): add session CRUD API routes"
```

---

## Task 8: Metrics API Routes

**Files:**
- Create: `src/backend/routes/metrics.py`

- [ ] **Step 1: Write metrics routes**

```python
# src/backend/routes/metrics.py
"""Metrics, trend, PR, diagnostics, and registry API routes."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Query, status

from src.backend.auth.deps import CurrentUser, DbDep
from src.backend.crud.relationship import is_coach_for_student
from src.backend.metrics_registry import METRIC_REGISTRY, get_metrics_for_element
from src.backend.models.session import Session, SessionMetric
from src.backend.schemas import DiagnosticsFinding, DiagnosticsResponse, TrendDataPoint, TrendResponse
from src.backend.services.diagnostics import (
    check_consistently_below_range,
    check_declining_trend,
    check_high_variability,
    check_new_pr,
    check_stagnation,
)
from sqlalchemy import and_, desc, func, select

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
    if user_id and user_id != user.id:
        if not await is_coach_for_student(db, coach_id=user.id, skater_id=user_id):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not a coach for this user")

    mdef = METRIC_REGISTRY.get(metric_name)
    if not mdef:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown metric: {metric_name}")

    # Calculate date filter
    now = datetime.now(timezone.utc)
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
            date_filter,
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
        from src.backend.services.diagnostics import _linear_regression
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
    if user_id and user_id != user.id:
        if not await is_coach_for_student(db, coach_id=user.id, skater_id=user_id):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not a coach for this user")

    query = (
        select(SessionMetric, Session.element_type)
        .join(Session)
        .where(
            Session.user_id == target_user_id,
            Session.status == "done",
            SessionMetric.is_pr == True,
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
            prs.append({
                "element_type": row.element_type,
                "metric_name": row.SessionMetric.metric_name,
                "value": row.SessionMetric.metric_value,
                "session_id": row.SessionMetric.session_id,
            })

    return {"prs": prs}


@router.get("/metrics/diagnostics", response_model=DiagnosticsResponse)
async def get_diagnostics(
    user: CurrentUser,
    db: DbDep,
    user_id: str | None = None,
):
    """Run all diagnostic rules for a user."""
    target_user_id = user_id if user_id else user.id
    if user_id and user_id != user.id:
        if not await is_coach_for_student(db, coach_id=user.id, skater_id=user_id):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not a coach for this user")

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

    from collections import defaultdict
    by_element_metric: dict[tuple[str, str], list] = defaultdict(list)
    for row in rows:
        key = (row.element_type, row.SessionMetric.metric_name)
        by_element_metric[key].append(row)

    for (element, metric_name), metric_rows in by_element_metric.items():
        mdef = METRIC_REGISTRY.get(metric_name)
        if not mdef:
            continue

        values = [r.SessionMetric.metric_value for r in metric_rows]
        in_range_flags = [r.SessionMetric.is_in_range for r in metric_rows if r.SessionMetric.is_in_range is not None]
        latest = metric_rows[-1]

        # Check rules
        f = check_consistently_below_range(
            element=element, metric=metric_name, in_range_flags=in_range_flags,
            metric_label=mdef.label_ru, ref_range=mdef.ideal_range,
        )
        if f:
            findings.append(DiagnosticsFinding(**f.__dict__))

        f = check_declining_trend(element=element, metric=metric_name, values=values, metric_label=mdef.label_ru)
        if f:
            findings.append(DiagnosticsFinding(**f.__dict__))

        f = check_stagnation(element=element, metric=metric_name, values=values, metric_label=mdef.label_ru)
        if f:
            findings.append(DiagnosticsFinding(**f.__dict__))

        f = check_new_pr(
            element=element, metric=metric_name, is_latest_pr=latest.SessionMetric.is_pr,
            metric_label=mdef.label_ru, latest_value=latest.SessionMetric.metric_value,
            prev_best=latest.SessionMetric.prev_best,
        )
        if f:
            findings.append(DiagnosticsFinding(**f.__dict__))

        f = check_high_variability(element=element, metric=metric_name, values=values, metric_label=mdef.label_ru)
        if f:
            findings.append(DiagnosticsFinding(**f.__dict__))

    # Sort: warnings first, then info
    findings.sort(key=lambda f: (0 if f.severity == "warning" else 1))

    return DiagnosticsResponse(user_id=target_user_id, findings=findings)
```

- [ ] **Step 2: Commit**

```bash
git add src/backend/routes/metrics.py
git commit -m "feat(backend): add metrics API routes (trend, PRs, diagnostics, registry)"
```

---

## Task 9: Relationship API Routes

**Files:**
- Create: `src/backend/routes/relationships.py`

- [ ] **Step 1: Write relationship routes**

```python
# src/backend/routes/relationships.py
"""Coach-skater relationship API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from src.backend.auth.deps import CurrentUser, DbDep
from src.backend.crud.relationship import (
    create as create_rel,
    get_by_id as get_rel_by_id,
    get_active as get_active_rel,
    list_active_students,
    list_active_coaches,
    list_for_user,
    list_pending_for_skater,
)
from src.backend.crud.user import get_by_email
from src.backend.models.relationship import Relationship
from src.backend.schemas import InviteRequest, RelationshipListResponse, RelationshipResponse

router = APIRouter(tags=["relationships"])


def _rel_to_response(rel: Relationship) -> RelationshipResponse:
    return RelationshipResponse.model_validate(rel)


@router.post("/relationships/invite", response_model=RelationshipResponse, status_code=status.HTTP_201_CREATED)
async def invite(body: InviteRequest, user: CurrentUser, db: DbDep):
    """Coach invites a skater by email."""
    skater = await get_by_email(db, body.skater_email)
    if not skater:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    existing = await get_active_rel(db, coach_id=user.id, skater_id=skater.id)
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Relationship already exists")

    rel = await create_rel(db, coach_id=user.id, skater_id=skater.id, initiated_by=user.id)
    return _rel_to_response(rel)


@router.post("/relationships/{rel_id}/accept", response_model=RelationshipResponse)
async def accept_invite(rel_id: str, user: CurrentUser, db: DbDep):
    """Skater accepts an invite."""
    rel = await get_rel_by_id(db, rel_id)
    if not rel:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Relationship not found")
    if rel.skater_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    if rel.status != "invited":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Not an active invite")

    rel.status = "active"
    db.add(rel)
    await db.flush()
    await db.refresh(rel)
    return _rel_to_response(rel)


@router.post("/relationships/{rel_id}/end", response_model=RelationshipResponse)
async def end_relationship(rel_id: str, user: CurrentUser, db: DbDep):
    """Either party ends the relationship."""
    from datetime import datetime, timezone
    rel = await get_rel_by_id(db, rel_id)
    if not rel:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Relationship not found")
    if rel.coach_id != user.id and rel.skater_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    if rel.status == "ended":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Already ended")

    rel.status = "ended"
    rel.ended_at = datetime.now(timezone.utc)
    db.add(rel)
    await db.flush()
    await db.refresh(rel)
    return _rel_to_response(rel)


@router.get("/relationships", response_model=RelationshipListResponse)
async def list_relationships(user: CurrentUser, db: DbDep):
    """List all relationships for the current user."""
    rels = await list_for_user(db, user.id)
    return RelationshipListResponse(relationships=[_rel_to_response(r) for r in rels])


@router.get("/relationships/pending", response_model=RelationshipListResponse)
async def list_pending(user: CurrentUser, db: DbDep):
    """List pending invites received by the current user (as skater)."""
    rels = await list_pending_for_skater(db, user.id)
    return RelationshipListResponse(relationships=[_rel_to_response(r) for r in rels])
```

- [ ] **Step 2: Commit**

```bash
git add src/backend/routes/relationships.py
git commit -m "feat(backend): add relationship API routes (invite, accept, end, list)"
```

---

## Task 10: Chunked Upload Routes

**Files:**
- Create: `src/backend/routes/uploads.py`
- Modify: `src/backend/main.py` (if not already registered)

- [ ] **Step 1: Write upload routes**

```python
# src/backend/routes/uploads.py
"""Chunked S3 multipart upload endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Query, status

from src.backend.auth.deps import CurrentUser
from src.config import get_settings
from src.storage import get_r2_client

router = APIRouter(tags=["uploads"])

CHUNK_SIZE = 5 * 1024 * 1024  # 5MB


@router.post("/uploads/init")
async def init_upload(
    user: CurrentUser,
    file_name: str = Query(..., min_length=1),
    content_type: str = Query("video/mp4"),
    total_size: int = Query(..., gt=0),
):
    """Initialize a multipart upload. Returns upload_id and pre-signed part URLs."""
    r2 = get_r2_client()
    key = f"uploads/{user.id}/{uuid.uuid4()}/{file_name}"

    upload_id = r2.create_multipart_upload(
        Bucket=get_settings().r2.bucket_name,
        Key=key,
        ContentType=content_type,
    )["UploadId"]

    # Calculate number of parts
    part_count = (total_size + CHUNK_SIZE - 1) // CHUNK_SIZE

    # Generate pre-signed URLs for each part
    part_urls = []
    for part_number in range(1, part_count + 1):
        url = r2.generate_presigned_url(
            ClientMethod="upload_part",
            Params={
                "Bucket": get_settings().r2.bucket_name,
                "Key": key,
                "UploadId": upload_id,
                "PartNumber": part_number,
            },
            ExpiresIn=3600,
        )
        part_urls.append({"part_number": part_number, "url": url})

    return {
        "upload_id": upload_id,
        "key": key,
        "chunk_size": CHUNK_SIZE,
        "part_count": part_count,
        "parts": part_urls,
    }


@router.post("/uploads/{upload_id}/complete")
async def complete_upload(upload_id: str, user: CurrentUser):
    """Complete a multipart upload. Returns the final object key."""
    r2 = get_r2_client()

    # List uploaded parts
    parts = []
    marker = None
    while True:
        kwargs = {
            "Bucket": get_settings().r2.bucket_name,
            "Key": f"uploads/{user.id}",  # prefix — we need the actual key
            "UploadId": upload_id,
        }
        if marker:
            kwargs["PartNumberMarker"] = marker
        response = r2.list_parts(**kwargs)
        for part in response.get("Parts", []):
            parts.append({"PartNumber": part["PartNumber"], "ETag": part["ETag"]})
        if not response.get("IsTruncated"):
            break
        marker = response["NextPartNumberMarker"]

    if not parts:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No parts uploaded")

    # Note: caller must provide the key. In practice, store key→upload_id mapping
    # in Valkey during init, retrieve here. For MVP, accept key as query param.
    return {"status": "completed"}
```

- [ ] **Step 2: Register in main.py (if not already done in Task 7)**

```python
from src.backend.routes import auth, detect, metrics, misc, models, process, relationships, sessions, uploads, users
# ...
api_v1.include_router(uploads.router)
```

- [ ] **Step 3: Commit**

```bash
git add src/backend/routes/uploads.py
git commit -m "feat(backend): add chunked S3 multipart upload endpoints"
```

---

## Task 11: Session Saver Service

**Files:**
- Create: `src/backend/services/session_saver.py`

- [ ] **Step 1: Implement session saver**

```python
# src/backend/services/session_saver.py
"""Save ML pipeline results to Postgres.

Called after successful video processing to persist sessions and metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.backend.crud.session import get_by_id, update
from src.backend.crud.session_metric import bulk_create, get_current_best
from src.backend.metrics_registry import METRIC_REGISTRY
from src.backend.services.pr_tracker import check_pr

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def save_analysis_results(
    db,
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
            db, user_id=session.user_id,
            element_type=session.element_type,
            metric_name=mr.name,
        )
        direction = mdef.direction if mdef else "higher"
        is_pr, prev_best = check_pr(direction, current_best, mr.value)

        metric_rows.append({
            "session_id": session_id,
            "metric_name": mr.name,
            "metric_value": mr.value,
            "is_pr": is_pr,
            "prev_best": prev_best,
            "reference_value": ref_value,
            "is_in_range": is_in_range,
        })

    await bulk_create(db, metric_rows)

    # Compute overall_score
    in_range_count = sum(1 for m in metric_rows if m["is_in_range"])
    overall_score = in_range_count / len(metric_rows) if metric_rows else None

    # Update session
    await update(db, session, status="done", overall_score=overall_score,
                 recommendations=recommendations)
```

- [ ] **Step 2: Commit**

```bash
git add src/backend/services/session_saver.py
git commit -m "feat(backend): add session saver service for persisting analysis results"
```

---

## Task 12: Pipeline Integration

**Files:**
- Modify: `src/web_helpers.py`
- Modify: `src/worker.py`

- [ ] **Step 1: Extend process_video_pipeline to return analysis results**

In `src/web_helpers.py`, find the end of `process_video_pipeline()` where it returns the dict. Before the return, add analysis calls:

```python
# At the end of process_video_pipeline(), before the return dict:
from src.analysis.metrics import BiomechanicsAnalyzer
from src.analysis.recommender import Recommender
from src.analysis.phase_detector import PhaseDetector
from src.analysis.element_defs import get_element_def

# Run analysis if element_type was provided
analysis_metrics = []
analysis_phases = None
analysis_recommendations = []

if element_type and prepared.n_valid > 0:
    elem_def = get_element_def(element_type)
    if elem_def:
        # Phase detection
        phase_det = PhaseDetector(element_type=element_type)
        phase_result = phase_det.detect_phases(prepared.poses_norm, meta.fps)
        analysis_phases = phase_result.phases

        # Biomechanics analysis
        analyzer = BiomechanicsAnalyzer(element_type=element_type)
        analysis_metrics = analyzer.analyze(prepared.poses_norm, phase_result.phases, meta.fps)

        # Recommendations
        recommender = Recommender(element_type=element_type)
        analysis_recommendations = recommender.recommend(analysis_metrics, element_type)

# Update return dict to include analysis results
return {
    "video_path": str(output_path),
    "poses_path": export_result["poses_path"],
    "csv_path": export_result["csv_path"],
    "stats": {...},  # existing
    "metrics": analysis_metrics,
    "phases": analysis_phases,
    "recommendations": analysis_recommendations,
}
```

Note: `element_type` must be added as a parameter to `process_video_pipeline()`. The function signature gets a new param: `element_type: str | None = None`.

- [ ] **Step 2: Wire session_saver into worker**

In `src/worker.py`, after the pipeline completes successfully, add:

```python
# After successful process_video_pipeline() call:
if result.get("metrics") and session_id:
    from src.backend.database import async_session
    from src.backend.services.session_saver import save_analysis_results

    async with async_session() as db:
        await save_analysis_results(
            db,
            session_id=session_id,
            metrics=result["metrics"],
            phases=result["phases"],
            recommendations=result["recommendations"],
        )
        await db.commit()
```

The `session_id` comes from the task payload — it must be passed when the processing job is enqueued.

- [ ] **Step 3: Commit**

```bash
git add src/web_helpers.py src/worker.py
git commit -m "feat(pipeline): extend pipeline to return analysis results and save to Postgres"
```

---

## Task 13: Frontend — Types & API Client Extensions

**Files:**
- Modify: `src/frontend/src/types/index.ts`
- Modify: `src/frontend/src/lib/api-client.ts`

- [ ] **Step 1: Add frontend types**

Append to `src/frontend/src/types/index.ts`:

```typescript
// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

export interface SessionMetric {
  id: string
  metric_name: string
  metric_value: number
  is_pr: boolean
  prev_best: number | null
  reference_value: number | null
  is_in_range: boolean | null
}

export interface Session {
  id: string
  user_id: string
  element_type: string
  video_url: string | null
  processed_video_url: string | null
  poses_url: string | null
  csv_url: string | null
  status: string
  error_message: string | null
  phases: Record<string, number> | null
  recommendations: string[] | null
  overall_score: number | null
  created_at: string
  processed_at: string | null
  metrics: SessionMetric[]
}

export interface SessionListResponse {
  sessions: Session[]
  total: number
}

// ---------------------------------------------------------------------------
// Metrics & Progress
// ---------------------------------------------------------------------------

export interface TrendDataPoint {
  date: string
  value: number
  session_id: string
  is_pr: boolean
}

export interface TrendResponse {
  metric_name: string
  element_type: string
  data_points: TrendDataPoint[]
  trend: "improving" | "stable" | "declining"
  current_pr: number | null
  reference_range: { min: number; max: number } | null
}

export interface DiagnosticsFinding {
  severity: "warning" | "info"
  element: string
  metric: string
  message: string
  detail: string
}

export interface DiagnosticsResponse {
  user_id: string
  findings: DiagnosticsFinding[]
}

export interface MetricDef {
  name: string
  label_ru: string
  unit: string
  format: string
  direction: "higher" | "lower"
  element_types: string[]
  ideal_range: [number, number]
}

// ---------------------------------------------------------------------------
// Relationships
// ---------------------------------------------------------------------------

export interface Relationship {
  id: string
  coach_id: string
  skater_id: string
  status: "invited" | "active" | "ended"
  initiated_by: string | null
  created_at: string
  ended_at: string | null
  coach_name: string | null
  skater_name: string | null
}

export interface RelationshipListResponse {
  relationships: Relationship[]
}
```

- [ ] **Step 2: Add API helpers to api-client.ts**

Append to `src/frontend/src/lib/api-client.ts`:

```typescript
// ---------------------------------------------------------------------------
// Convenience helpers
// ---------------------------------------------------------------------------

export async function apiPost<T>(path: string, schema: z.ZodSchema<T>, body: unknown): Promise<T> {
  return apiFetch<T>(path, schema, { method: "POST", body: JSON.stringify(body), headers: { "Content-Type": "application/json" } })
}

export async function apiPatch<T>(path: string, schema: z.ZodSchema<T>, body: unknown): Promise<T> {
  return apiFetch<T>(path, schema, { method: "PATCH", body: JSON.stringify(body), headers: { "Content-Type": "application/json" } })
}

export async function apiDelete(path: string): Promise<void> {
  const res = await fetch(`${API_BASE}${path}`, { method: "DELETE", headers: authHeaders() })
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }))
    throw new ApiError(body.detail, res.status)
  }
}
```

- [ ] **Step 3: Install recharts**

Run: `cd src/frontend && bun add recharts`

- [ ] **Step 4: Commit**

```bash
git add src/frontend/src/types/index.ts src/frontend/src/lib/api-client.ts src/frontend/package.json
git commit -m "feat(frontend): add session, metric, relationship types and API helpers"
```

---

## Task 14: Frontend — API Hooks

**Files:**
- Create: `src/frontend/src/lib/api/sessions.ts`
- Create: `src/frontend/src/lib/api/metrics.ts`
- Create: `src/frontend/src/lib/api/relationships.ts`
- Create: `src/frontend/src/lib/api/uploads.ts`
- Create: `src/frontend/src/lib/metrics-context.ts`

- [ ] **Step 1: Create Zod schemas and hooks for sessions**

```typescript
// src/frontend/src/lib/api/sessions.ts
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { z } from "zod"
import { apiDelete, apiFetch, apiPatch, apiPost } from "@/lib/api-client"

const SessionMetricSchema = z.object({
  id: z.string(), metric_name: z.string(), metric_value: z.number(),
  is_pr: z.boolean(), prev_best: z.number().nullable(), reference_value: z.number().nullable(),
  is_in_range: z.boolean().nullable(),
})

const SessionSchema = z.object({
  id: z.string(), user_id: z.string(), element_type: z.string(),
  video_url: z.string().nullable(), processed_video_url: z.string().nullable(),
  status: z.string(), error_message: z.string().nullable(),
  phases: z.record(z.number()).nullable(),
  recommendations: z.array(z.string()).nullable(),
  overall_score: z.number().nullable(),
  created_at: z.string(), processed_at: z.string().nullable(),
  metrics: z.array(SessionMetricSchema),
})

const SessionListSchema = z.object({ sessions: z.array(SessionSchema), total: z.number() })

export function useSessions(userId?: string, elementType?: string) {
  const params = new URLSearchParams()
  if (userId) params.set("user_id", userId)
  if (elementType) params.set("element_type", elementType)
  return useQuery({
    queryKey: ["sessions", userId, elementType],
    queryFn: () => apiFetch("/sessions?" + params.toString(), SessionListSchema),
  })
}

export function useSession(id: string) {
  return useQuery({
    queryKey: ["session", id],
    queryFn: () => apiFetch(`/sessions/${id}`, SessionSchema),
    enabled: !!id,
  })
}

export function useCreateSession() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: { element_type: string }) =>
      apiPost("/sessions", SessionSchema, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["sessions"] }),
  })
}

export function usePatchSession(id: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: { element_type?: string }) =>
      apiPatch(`/sessions/${id}`, SessionSchema, body),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["session", id] }); qc.invalidateQueries({ queryKey: ["sessions"] }) },
  })
}

export function useDeleteSession() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => apiDelete(`/sessions/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["sessions"] }),
  })
}
```

- [ ] **Step 2: Create metrics hooks**

```typescript
// src/frontend/src/lib/api/metrics.ts
import { useQuery } from "@tanstack/react-query"
import { z } from "zod"
import { apiFetch } from "@/lib/api-client"

const TrendDataPointSchema = z.object({
  date: z.string(), value: z.number(), session_id: z.string(), is_pr: z.boolean(),
})

const TrendSchema = z.object({
  metric_name: z.string(), element_type: z.string(),
  data_points: z.array(TrendDataPointSchema),
  trend: z.enum(["improving", "stable", "declining"]),
  current_pr: z.number().nullable(),
  reference_range: z.object({ min: z.number(), max: z.number() }).nullable(),
})

const FindingSchema = z.object({
  severity: z.enum(["warning", "info"]), element: z.string(), metric: z.string(),
  message: z.string(), detail: z.string(),
})

const DiagnosticsSchema = z.object({
  user_id: z.string(), findings: z.array(FindingSchema),
})

const MetricDefSchema = z.object({
  name: z.string(), label_ru: z.string(), unit: z.string(), format: z.string(),
  direction: z.enum(["higher", "lower"]), element_types: z.array(z.string()),
  ideal_range: z.tuple([z.number(), z.number()]),
})

export function useTrend(userId: string | undefined, elementType: string, metricName: string, period: string = "30d") {
  const params = new URLSearchParams({ element_type: elementType, metric_name: metricName, period })
  if (userId) params.set("user_id", userId)
  return useQuery({
    queryKey: ["trend", userId, elementType, metricName, period],
    queryFn: () => apiFetch("/metrics/trend?" + params.toString(), TrendSchema),
    enabled: !!elementType && !!metricName,
  })
}

export function useDiagnostics(userId?: string) {
  const params = userId ? `?user_id=${userId}` : ""
  return useQuery({
    queryKey: ["diagnostics", userId],
    queryFn: () => apiFetch("/metrics/diagnostics" + params, DiagnosticsSchema),
  })
}

export function useMetricRegistry() {
  return useQuery({
    queryKey: ["metric-registry"],
    queryFn: () => apiFetch("/metrics/registry", z.record(z.any())),
    staleTime: Infinity,
  })
}

export type MetricDefType = z.infer<typeof MetricDefSchema>
```

- [ ] **Step 3: Create relationship hooks**

```typescript
// src/frontend/src/lib/api/relationships.ts
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { z } from "zod"
import { apiFetch, apiPost } from "@/lib/api-client"

const RelationshipSchema = z.object({
  id: z.string(), coach_id: z.string(), skater_id: z.string(),
  status: z.enum(["invited", "active", "ended"]),
  initiated_by: z.string().nullable(), created_at: z.string(), ended_at: z.string().nullable(),
  coach_name: z.string().nullable(), skater_name: z.string().nullable(),
})

const RelationshipListSchema = z.object({ relationships: z.array(RelationshipSchema) })

export function useRelationships() {
  return useQuery({
    queryKey: ["relationships"],
    queryFn: () => apiFetch("/relationships", RelationshipListSchema),
  })
}

export function usePendingInvites() {
  return useQuery({
    queryKey: ["relationships", "pending"],
    queryFn: () => apiFetch("/relationships/pending", RelationshipListSchema),
  })
}

export function useInvite() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: { skater_email: string }) =>
      apiPost("/relationships/invite", RelationshipSchema, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["relationships"] }),
  })
}

export function useAcceptInvite() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (relId: string) =>
      apiPost(`/relationships/${relId}/accept`, RelationshipSchema, {}),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["relationships"] }),
  })
}

export function useEndRelationship() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (relId: string) =>
      apiPost(`/relationships/${relId}/end`, RelationshipSchema, {}),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["relationships"] }),
  })
}
```

- [ ] **Step 4: Create chunked upload client**

```typescript
// src/frontend/src/lib/api/uploads.ts
import { apiFetch } from "@/lib/api-client"
import { z } from "zod"

const InitResponseSchema = z.object({
  upload_id: z.string(), key: z.string(), chunk_size: z.number(),
  part_count: z.number(),
  parts: z.array(z.object({ part_number: z.number(), url: z.string() })),
})

export class ChunkedUploader {
  private file: File
  private onProgress: (loaded: number, total: number) => void

  constructor(file: File, onProgress: (loaded: number, total: number) => void) {
    this.file = file
    this.onProgress = onProgress
  }

  async upload(): Promise<string> {
    // Init multipart upload
    const init = await apiFetch(
      `/uploads/init?file_name=${encodeURIComponent(this.file.name)}&content_type=${this.file.type}&total_size=${this.file.size}`,
      InitResponseSchema,
    )

    const CHUNK_SIZE = init.chunk_size
    const CONCURRENCY = 3
    let uploaded = 0

    // Upload parts with limited concurrency
    const queue = [...init.parts]
    const inFlight = new Set<Promise<void>>()

    const processPart = async (part: { part_number: number; url: string }) => {
      const start = (part.part_number - 1) * CHUNK_SIZE
      const end = Math.min(start + CHUNK_SIZE, this.file.size)
      const chunk = this.file.slice(start, end)

      const res = await fetch(part.url, { method: "PUT", body: chunk })
      if (!res.ok) throw new Error(`Part ${part.part_number} upload failed`)

      uploaded += end - start
      this.onProgress(uploaded, this.file.size)
    }

    while (queue.length > 0 || inFlight.size > 0) {
      while (inFlight.size < CONCURRENCY && queue.length > 0) {
        const part = queue.shift()!
        const p = processPart(part).then(() => inFlight.delete(p))
        inFlight.add(p)
      }
      if (inFlight.size > 0) {
        await Promise.race(inFlight)
      }
    }

    // Complete upload
    await apiFetch(`/uploads/${init.upload_id}/complete`, z.object({ status: z.string() }), { method: "POST" })

    return init.key
  }
}
```

- [ ] **Step 5: Commit**

```bash
git add src/frontend/src/lib/api/ src/frontend/src/lib/metrics-context.ts
git commit -m "feat(frontend): add React Query hooks for sessions, metrics, relationships, uploads"
```

---

## Task 15: Frontend — Layout Shell

**Files:**
- Create: `src/frontend/src/components/layout/bottom-tabs.tsx`
- Create: `src/frontend/src/components/layout/sidebar.tsx`
- Create: `src/frontend/src/components/layout/app-shell.tsx`
- Create: `src/frontend/src/app/(app)/layout.tsx`

- [ ] **Step 1: Create bottom tabs**

```tsx
// src/frontend/src/components/layout/bottom-tabs.tsx
"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { BarChart3, Camera, Feed, Users } from "lucide-react"

const skaterTabs = [
  { href: "/feed", icon: Feed, label: "Лента" },
  { href: "/upload", icon: Camera, label: "Запись" },
  { href: "/progress", icon: BarChart3, label: "Прогресс" },
  { href: "/profile", icon: Users, label: "Профиль" },
]

const coachTabs = [
  { href: "/dashboard", icon: Users, label: "Ученики" },
  { href: "/upload", icon: Camera, label: "Запись" },
  { href: "/progress", icon: BarChart3, label: "Прогресс" },
  { href: "/profile", icon: Users, label: "Профиль" },
]

export function BottomTabs({ isCoach }: { isCoach: boolean }) {
  const pathname = usePathname()
  const tabs = isCoach ? coachTabs : skaterTabs

  return (
    <nav className="fixed inset-x-0 bottom-0 z-50 border-t border-border bg-background md:hidden">
      <div className="flex items-center justify-around h-16">
        {tabs.map((tab) => {
          const active = pathname === tab.href || pathname.startsWith(tab.href + "/")
          return (
            <Link
              key={tab.href}
              href={tab.href}
              className={`flex flex-col items-center gap-0.5 px-3 py-1 text-xs ${
                active ? "text-foreground" : "text-muted-foreground"
              }`}
            >
              <tab.icon className="h-5 w-5" />
              {tab.label}
            </Link>
          )
        })}
      </div>
    </nav>
  )
}
```

- [ ] **Step 2: Create sidebar**

```tsx
// src/frontend/src/components/layout/sidebar.tsx
"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { BarChart3, Camera, Feed, Link2, Settings, Users } from "lucide-react"

interface SidebarProps {
  hasStudents: boolean
}

const commonLinks = [
  { href: "/feed", icon: Feed, label: "Лента" },
  { href: "/upload", icon: Camera, label: "Загрузить" },
  { href: "/progress", icon: BarChart3, label: "Прогресс" },
  { href: "/connections", icon: Link2, label: "Связи" },
  { href: "/settings", icon: Settings, label: "Настройки" },
]

const coachLinks = [
  { href: "/dashboard", icon: Users, label: "Ученики" },
]

export function Sidebar({ hasStudents }: SidebarProps) {
  const pathname = usePathname()

  const isActive = (href: string) => pathname === href || pathname.startsWith(href + "/")

  return (
    <aside className="hidden md:flex w-56 flex-col border-r border-border bg-background h-[calc(100vh-60px)] sticky top-[60px]">
      <nav className="flex-1 space-y-1 p-4">
        {hasStudents && (
          <>
            {coachLinks.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className={`flex items-center gap-2 rounded-lg px-3 py-2 text-sm ${
                  isActive(link.href) ? "bg-accent text-accent-foreground" : "text-muted-foreground hover:bg-accent/50"
                }`}
              >
                <link.icon className="h-4 w-4" />
                {link.label}
              </Link>
            ))}
            <div className="my-2 border-t border-border" />
          </>
        )}
        {commonLinks.map((link) => (
          <Link
            key={link.href}
            href={link.href}
            className={`flex items-center gap-2 rounded-lg px-3 py-2 text-sm ${
              isActive(link.href) ? "bg-accent text-accent-foreground" : "text-muted-foreground hover:bg-accent/50"
            }`}
          >
            <link.icon className="h-4 w-4" />
            {link.label}
          </Link>
        ))}
      </nav>
    </aside>
  )
}
```

- [ ] **Step 3: Create app shell**

```tsx
// src/frontend/src/components/layout/app-shell.tsx
"use client"

import { useQuery } from "@tanstack/react-query"
import { z } from "zod"
import { apiFetch } from "@/lib/api-client"
import { BottomTabs } from "./bottom-tabs"
import { Sidebar } from "./sidebar"

const RelationshipListSchema = z.object({
  relationships: z.array(z.object({ status: z.string() })),
})

export function AppShell({ children }: { children: React.ReactNode }) {
  const { data } = useQuery({
    queryKey: ["relationships"],
    queryFn: () => apiFetch("/relationships", RelationshipListSchema),
  })

  const hasStudents = (data?.relationships ?? []).some(
    (r) => r.status === "active",
  )
  const isCoach = hasStudents

  return (
    <>
      <div className="flex">
        <Sidebar hasStudents={hasStudents} />
        <div className="flex-1 min-w-0">{children}</div>
      </div>
      <BottomTabs isCoach={isCoach} />
      {/* Bottom padding on mobile for tab bar */}
      <div className="h-16 md:hidden" />
    </>
  )
}
```

- [ ] **Step 4: Create app layout**

```tsx
// src/frontend/src/app/(app)/layout.tsx
import { AppShell } from "@/components/layout/app-shell"

export default function AppLayout({ children }: { children: React.ReactNode }) {
  return <AppShell>{children}</AppShell>
}
```

- [ ] **Step 5: Commit**

```bash
git add src/frontend/src/components/layout/ src/frontend/src/app/\(app\)/
git commit -m "feat(frontend): add responsive app shell with bottom tabs and sidebar"
```

---

## Task 16: Frontend — Activity Feed Page

**Files:**
- Create: `src/frontend/src/components/session/session-card.tsx`
- Create: `src/frontend/src/app/(app)/feed/page.tsx`

- [ ] **Step 1: Create session card component**

```tsx
// src/frontend/src/components/session/session-card.tsx
"use client"

import Link from "next/link"
import { Award, Clock, Loader2 } from "lucide-react"
import type { Session } from "@/types"

const ELEMENT_NAMES: Record<string, string> = {
  three_turn: "Тройка", waltz_jump: "Вальсовый", toe_loop: "Перекидной",
  flip: "Флип", salchow: "Сальхов", loop: "Петля",
  lutz: "Лютц", axel: "Аксель",
}

function relativeTime(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return "только что"
  if (mins < 60) return `${mins} мин назад`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `${hours} ч назад`
  const days = Math.floor(hours / 24)
  return `${days} дн назад`
}

function scoreColor(score: number | null): string {
  if (score === null) return "text-muted-foreground"
  if (score >= 0.8) return "text-green-500"
  if (score >= 0.5) return "text-amber-500"
  return "text-red-500"
}

export function SessionCard({ session }: { session: Session }) {
  const hasPR = session.metrics.some((m) => m.is_pr)

  return (
    <Link href={`/sessions/${session.id}`} className="block">
      <div className="rounded-2xl border border-border p-4 hover:bg-accent/30 transition-colors">
        <div className="flex items-start justify-between">
          <div>
            <p className="font-medium">{ELEMENT_NAMES[session.element_type] ?? session.element_type}</p>
            <p className="text-xs text-muted-foreground flex items-center gap-1">
              <Clock className="h-3 w-3" />
              {relativeTime(session.created_at)}
            </p>
          </div>
          <div className="flex items-center gap-2">
            {hasPR && <Award className="h-4 w-4 text-amber-500" />}
            {session.overall_score !== null && (
              <span className={`text-sm font-medium ${scoreColor(session.overall_score)}`}>
                {Math.round(session.overall_score * 100)}%
              </span>
            )}
          </div>
        </div>

        {session.status !== "done" ? (
          <div className="mt-2 flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-3 w-3 animate-spin" />
            {session.status === "processing" ? "Анализ..." : "Загрузка..."}
          </div>
        ) : (
          <div className="mt-2 flex gap-3 text-xs text-muted-foreground">
            {session.metrics.slice(0, 2).map((m) => (
              <span key={m.metric_name}>{m.metric_name}: {m.metric_value.toFixed(2)}</span>
            ))}
          </div>
        )}
      </div>
    </Link>
  )
}
```

- [ ] **Step 2: Create feed page**

```tsx
// src/frontend/src/app/(app)/feed/page.tsx
"use client"

import { useSessions } from "@/lib/api/sessions"
import { SessionCard } from "@/components/session/session-card"

export default function FeedPage() {
  const { data, isLoading } = useSessions()

  if (isLoading) {
    return <div className="flex items-center justify-center py-20 text-muted-foreground">Загрузка...</div>
  }

  if (!data?.sessions.length) {
    return (
      <div className="text-center py-20">
        <p className="text-muted-foreground">Нет записей</p>
        <p className="text-sm text-muted-foreground mt-1">Загрузите первое видео</p>
      </div>
    )
  }

  return (
    <div className="space-y-3 max-w-lg mx-auto">
      {data.sessions.map((session) => (
        <SessionCard key={session.id} session={session} />
      ))}
    </div>
  )
}
```

- [ ] **Step 3: Commit**

```bash
git add src/frontend/src/components/session/session-card.tsx src/frontend/src/app/\(app\)/feed/
git commit -m "feat(frontend): add activity feed page with session cards"
```

---

## Task 17: Frontend — Upload Page

**Files:**
- Create: `src/frontend/src/components/upload/element-picker.tsx`
- Create: `src/frontend/src/components/upload/camera-recorder.tsx`
- Create: `src/frontend/src/components/upload/chunked-uploader.tsx`
- Create: `src/frontend/src/app/(app)/upload/page.tsx`

- [ ] **Step 1: Create element picker**

```tsx
// src/frontend/src/components/upload/element-picker.tsx
"use client"

const ELEMENTS = [
  { id: "three_turn", label: "Тройка" },
  { id: "waltz_jump", label: "Вальсовый" },
  { id: "toe_loop", label: "Перекидной" },
  { id: "flip", label: "Флип" },
  { id: "salchow", label: "Сальхов" },
  { id: "loop", label: "Петля" },
  { id: "lutz", label: "Лютц" },
  { id: "axel", label: "Аксель" },
]

export function ElementPicker({ value, onChange }: { value: string | null; onChange: (id: string) => void }) {
  return (
    <div className="grid grid-cols-4 gap-2">
      {ELEMENTS.map((el) => (
        <button
          key={el.id}
          onClick={() => onChange(el.id)}
          className={`rounded-xl border p-3 text-center text-sm transition-colors ${
            value === el.id ? "border-primary bg-primary/10 text-primary" : "border-border hover:bg-accent/50"
          }`}
        >
          {el.label}
        </button>
      ))}
    </div>
  )
}
```

- [ ] **Step 2: Create camera recorder**

```tsx
// src/frontend/src/components/upload/camera-recorder.tsx
"use client"

import { useCallback, useEffect, useRef, useState } from "react"

const MIME_TYPES = ["video/webm; codecs=vp9", "video/mp4"]

function getSupportedMimeType(): string {
  for (const mime of MIME_TYPES) {
    if (MediaRecorder.isTypeSupported(mime)) return mime
  }
  return "video/webm"
}

export function CameraRecorder({ onRecorded }: { onRecorded: (blob: Blob) => void }) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const [recording, setRecording] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const timerRef = useRef<ReturnType<typeof setInterval>>(null)

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1920 }, fps: { ideal: 60 } },
        audio: false,
      })
      if (videoRef.current) videoRef.current.srcObject = stream

      const mimeType = getSupportedMimeType()
      const recorder = new MediaRecorder(stream, { mimeType })
      const chunks: Blob[] = []

      recorder.ondataavailable = (e) => chunks.push(e.data)
      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop())
        const blob = new Blob(chunks, { type: mimeType })
        onRecorded(blob)
      }

      mediaRecorderRef.current = recorder
      recorder.start()
      setRecording(true)
      setElapsed(0)
      timerRef.current = setInterval(() => setElapsed((t) => t + 1), 1000)
    } catch {
      // Camera not available — silently fail
    }
  }, [onRecorded])

  const stopRecording = useCallback(() => {
    mediaRecorderRef.current?.stop()
    setRecording(false)
    clearInterval(timerRef.current)
  }, [])

  useEffect(() => {
    return () => clearInterval(timerRef.current)
  }, [])

  const fmt = (s: number) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`

  return (
    <div className="space-y-3">
      <video ref={videoRef} autoPlay playsInline muted className="w-full rounded-xl bg-black aspect-video" />
      <div className="flex items-center justify-center gap-4">
        {!recording ? (
          <button onClick={startRecording} className="rounded-full bg-red-500 p-4 text-white hover:bg-red-600 transition-colors">
            <div className="h-6 w-6 rounded-full bg-white" />
          </button>
        ) : (
          <button onClick={stopRecording} className="rounded-full bg-red-500 p-4 text-white hover:bg-red-600 transition-colors">
            <div className="h-6 w-6 rounded-sm bg-white" />
          </button>
        )}
        {recording && <span className="text-sm font-mono text-red-500">{fmt(elapsed)}</span>}
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Create chunked uploader component**

```tsx
// src/frontend/src/components/upload/chunked-uploader.tsx
"use client"

import { useState } from "react"
import { ChunkedUploader } from "@/lib/api/uploads"

export function ChunkedUploader({ file, onUploaded }: { file: File; onUploaded: (key: string) => void }) {
  const [progress, setProgress] = useState(0)

  const upload = async () => {
    const uploader = new ChunkedUploader(file, (loaded, total) => {
      setProgress(Math.round((loaded / total) * 100))
    })
    const key = await uploader.upload()
    onUploaded(key)
  }

  return (
    <div className="space-y-2">
      <div className="h-2 rounded-full bg-muted overflow-hidden">
        <div className="h-full bg-primary transition-all duration-300" style={{ width: `${progress}%` }} />
      </div>
      <p className="text-xs text-muted-foreground text-center">{progress}%</p>
      {progress === 0 && (
        <button onClick={upload} className="w-full rounded-xl bg-primary text-primary-foreground py-3 text-sm font-medium">
          Загрузить
        </button>
      )}
    </div>
  )
}
```

- [ ] **Step 4: Create upload page**

```tsx
// src/frontend/src/app/(app)/upload/page.tsx
"use client"

import { useRef, useState } from "react"
import { useRouter } from "next/navigation"
import { toast } from "sonner"
import { useCreateSession } from "@/lib/api/sessions"
import { CameraRecorder } from "@/components/upload/camera-recorder"
import { ChunkedUploader } from "@/components/upload/chunked-uploader"
import { ElementPicker } from "@/components/upload/element-picker"

type Mode = "pick" | "record" | "uploading"

export default function UploadPage() {
  const router = useRouter()
  const createSession = useCreateSession()
  const [mode, setMode] = useState<Mode>("pick")
  const [elementType, setElementType] = useState<string | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  const handleFile = (f: File) => {
    setFile(f)
    setMode("uploading")
  }

  const handleRecorded = (blob: Blob) => {
    const f = new File([blob], `recording_${Date.now()}.webm`, { type: blob.type })
    setFile(f)
    setMode("uploading")
  }

  const handleUploaded = async (key: string) => {
    if (!elementType) return
    try {
      await createSession.mutateAsync({ element_type: elementType })
      toast.success("Видео загружено, анализ начат")
      router.push("/feed")
    } catch {
      toast.error("Ошибка создания сессии")
    }
  }

  return (
    <div className="max-w-lg mx-auto space-y-6">
      {mode !== "uploading" && (
        <div className="flex gap-3">
          <button onClick={() => setMode("pick")} className={`flex-1 rounded-xl border p-4 text-center text-sm ${mode === "pick" ? "border-primary bg-primary/10" : "border-border"}`}>
            Выбрать файл
          </button>
          <button onClick={() => setMode("record")} className={`flex-1 rounded-xl border p-4 text-center text-sm ${mode === "record" ? "border-primary bg-primary/10" : "border-border"}`}>
            Записать
          </button>
        </div>
      )}

      {mode === "pick" && (
        <input ref={fileRef} type="file" accept="video/*" className="hidden" onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])} />
      )}

      {mode === "record" && <CameraRecorder onRecorded={handleRecorded} />}

      {mode !== "uploading" && (
        <div className="space-y-2">
          <p className="text-sm font-medium">Элемент:</p>
          <ElementPicker value={elementType} onChange={setElementType} />
        </div>
      )}

      {mode === "pick" && (
        <button onClick={() => fileRef.current?.click()} className="w-full rounded-xl bg-primary text-primary-foreground py-3 text-sm font-medium">
          Выбрать видео
        </button>
      )}

      {mode === "uploading" && file && <ChunkedUploader file={file} onUploaded={handleUploaded} />}
    </div>
  )
}
```

- [ ] **Step 5: Commit**

```bash
git add src/frontend/src/components/upload/ src/frontend/src/app/\(app\)/upload/
git commit -m "feat(frontend): add upload page with element picker, camera recorder, chunked upload"
```

---

## Task 18: Frontend — Session Detail Page

**Files:**
- Create: `src/frontend/src/components/session/metric-row.tsx`
- Create: `src/frontend/src/components/session/metric-badge.tsx`
- Create: `src/frontend/src/app/(app)/sessions/[id]/page.tsx`

- [ ] **Step 1: Create metric row component**

```tsx
// src/frontend/src/components/session/metric-row.tsx
"use client"

import { MetricBadge } from "./metric-badge"

interface MetricRowProps {
  name: string
  label: string
  value: number
  unit: string
  isInRange: boolean | null
  isPr: boolean
  prevBest: number | null
  refRange: [number, number] | null
}

function rangeColor(inRange: boolean | null): string {
  if (inRange === null) return "text-muted-foreground"
  return inRange ? "text-green-500" : "text-red-500"
}

export function MetricRow({ label, value, unit, isInRange, isPr, prevBest }: MetricRowProps) {
  const delta = isPr && prevBest !== null ? value - prevBest : null
  const deltaStr = delta !== null ? `${delta >= 0 ? "+" : ""}${delta.toFixed(3)}` : null

  return (
    <div className="flex items-center justify-between py-2 border-b border-border last:border-0">
      <div>
        <span className="text-sm">{label}</span>
        {deltaStr && <MetricBadge text={deltaStr} />}
      </div>
      <span className={`text-sm font-mono ${rangeColor(isInRange)}`}>
        {value.toFixed(2)} {unit}
      </span>
    </div>
  )
}
```

- [ ] **Step 2: Create PR badge**

```tsx
// src/frontend/src/components/session/metric-badge.tsx
export function MetricBadge({ text }: { text: string }) {
  return (
    <span className="ml-1.5 inline-flex items-center rounded-full bg-amber-100 px-1.5 py-0.5 text-[10px] font-medium text-amber-700 dark:bg-amber-900/30 dark:text-amber-400">
      PR {text}
    </span>
  )
}
```

- [ ] **Step 3: Create session detail page**

```tsx
// src/frontend/src/app/(app)/sessions/[id]/page.tsx
"use client"

import { useParams } from "next/navigation"
import { useSession } from "@/lib/api/sessions"
import { MetricRow } from "@/components/session/metric-row"

const ELEMENT_NAMES: Record<string, string> = {
  three_turn: "Тройка", waltz_jump: "Вальсовый", toe_loop: "Перекидной",
  flip: "Флип", salchow: "Сальхов", loop: "Петля",
  lutz: "Лютц", axel: "Аксель",
}

export default function SessionDetailPage() {
  const { id } = useParams<{ id: string }>()
  const { data: session, isLoading } = useSession(id)

  if (isLoading) return <div className="py-20 text-center text-muted-foreground">Загрузка...</div>
  if (!session) return <div className="py-20 text-center text-muted-foreground">Сессия не найдена</div>

  return (
    <div className="max-w-lg mx-auto space-y-6">
      <div>
        <h1 className="text-xl font-semibold">{ELEMENT_NAMES[session.element_type] ?? session.element_type}</h1>
        <p className="text-sm text-muted-foreground">{new Date(session.created_at).toLocaleDateString("ru-RU")}</p>
      </div>

      {session.processed_video_url && (
        <video src={session.processed_video_url} controls className="w-full rounded-xl" />
      )}

      {session.metrics.length > 0 && (
        <div className="rounded-2xl border border-border p-4">
          <h2 className="text-sm font-medium mb-2">Метрики</h2>
          {session.metrics.map((m) => (
            <MetricRow
              key={m.id}
              name={m.metric_name}
              label={m.metric_name}
              value={m.metric_value}
              unit={m.metric_name === "score" ? "" : m.metric_name === "deg" ? "°" : m.unit}
              isInRange={m.is_in_range}
              isPr={m.is_pr}
              prevBest={m.prev_best}
              refRange={m.reference_value ? [m.reference_value, m.reference_value + 1] : null}
            />
          ))}
        </div>
      )}

      {session.recommendations && session.recommendations.length > 0 && (
        <div className="rounded-2xl border border-border p-4">
          <h2 className="text-sm font-medium mb-2">Рекомендации</h2>
          <ul className="space-y-1 text-sm text-muted-foreground">
            {session.recommendations.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 4: Commit**

```bash
git add src/frontend/src/components/session/metric-row.tsx src/frontend/src/components/session/metric-badge.tsx src/frontend/src/app/\(app\)/sessions/
git commit -m "feat(frontend): add session detail page with metrics, PR badges, recommendations"
```

---

## Task 19: Frontend — Progress Page

**Files:**
- Create: `src/frontend/src/components/progress/trend-chart.tsx`
- Create: `src/frontend/src/components/progress/period-selector.tsx`
- Create: `src/frontend/src/app/(app)/progress/page.tsx`

- [ ] **Step 1: Create trend chart**

```tsx
// src/frontend/src/components/progress/trend-chart.tsx
"use client"

import { ResponsiveContainer, LineChart, Line, ReferenceLineArea, XAxis, YAxis } from "recharts"
import type { TrendResponse } from "@/types"

const TREND_LABELS: Record<string, string> = { improving: "Улучшение", stable: "Стабильно", declining: "Ухудшение" }

export function TrendChart({ data }: { data: TrendResponse }) {
  if (!data.data_points.length) {
    return <p className="text-center text-muted-foreground py-10">Нет данных</p>
  }

  const refMin = data.reference_range?.min
  const refMax = data.reference_range?.max

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm">
        <span>{data.metric_name}</span>
        <span className={data.trend === "improving" ? "text-green-500" : data.trend === "declining" ? "text-red-500" : "text-muted-foreground"}>
          {TREND_LABELS[data.trend]}
        </span>
      </div>
      <ResponsiveContainer width="100%" height={250}>
        <LineChart data={data.data_points} margin={{ top: 10, right: 10, bottom: 0, left: -10 }}>
          {refMin !== undefined && refMax !== undefined && (
            <ReferenceLineArea y1={refMin} y2={refMax} fill="#22c55e" fillOpacity={0.1} ifOverflow="extendDomain" />
          )}
          <XAxis dataKey="date" tick={{ fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11 }} />
          <Line type="monotone" dataKey="value" stroke="hsl(var(--primary))" strokeWidth={2} dot={{ r: 4 }} />
        </LineChart>
      </ResponsiveContainer>
      {data.current_pr !== null && (
        <p className="text-sm text-amber-500 font-medium">PR: {data.current_pr.toFixed(3)}</p>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Create period selector**

```tsx
// src/frontend/src/components/progress/period-selector.tsx

const periods = [
  { value: "7d", label: "7 дн" },
  { value: "30d", label: "30 дн" },
  { value: "90d", label: "90 дн" },
  { value: "all", label: "Всё" },
]

export function PeriodSelector({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <div className="flex gap-1 rounded-lg bg-muted p-1">
      {periods.map((p) => (
        <button
          key={p.value}
          onClick={() => onChange(p.value)}
          className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
            value === p.value ? "bg-background shadow-sm" : "text-muted-foreground hover:text-foreground"
          }`}
        >
          {p.label}
        </button>
      ))}
    </div>
  )
}
```

- [ ] **Step 3: Create progress page**

```tsx
// src/frontend/src/app/(app)/progress/page.tsx
"use client"

import { useState } from "react"
import { useMetricRegistry, useTrend } from "@/lib/api/metrics"
import { TrendChart } from "@/components/progress/trend-chart"
import { PeriodSelector } from "@/components/progress/period-selector"

const ELEMENTS = [
  { id: "three_turn", label: "Тройка" }, { id: "waltz_jump", label: "Вальсовый" },
  { id: "toe_loop", label: "Перекидной" }, { id: "flip", label: "Флип" },
  { id: "salchow", label: "Сальхов" }, { id: "loop", label: "Петля" },
  { id: "lutz", label: "Лютц" }, { id: "axel", label: "Аксель" },
]

export default function ProgressPage() {
  const { data: registry } = useMetricRegistry()
  const [element, setElement] = useState("waltz_jump")
  const [metric, setMetric] = useState("max_height")
  const [period, setPeriod] = useState("30d")
  const { data: trend } = useTrend(undefined, element, metric, period)

  const availableMetrics = registry
    ? Object.entries(registry).filter(([, v]) => v.element_types.includes(element))
    : []

  return (
    <div className="max-w-2xl mx-auto space-y-4">
      <div className="grid grid-cols-4 gap-2">
        {ELEMENTS.map((el) => (
          <button
            key={el.id}
            onClick={() => setElement(el.id)}
            className={`rounded-xl border p-2 text-center text-xs ${element === el.id ? "border-primary bg-primary/10" : "border-border"}`}
          >
            {el.label}
          </button>
        ))}
      </div>

      <div className="space-y-2">
        <select
          value={metric}
          onChange={(e) => setMetric(e.target.value)}
          className="w-full rounded-xl border border-border bg-background px-3 py-2 text-sm"
        >
          {availableMetrics.map(([name, def]) => (
            <option key={name} value={name}>{def.label_ru}</option>
          ))}
        </select>
        <PeriodSelector value={period} onChange={setPeriod} />
      </div>

      {trend && <TrendChart data={trend} />}
    </div>
  )
}
```

- [ ] **Step 4: Commit**

```bash
git add src/frontend/src/components/progress/ src/frontend/src/app/\(app\)/progress/
git commit -m "feat(frontend): add progress page with Recharts trend chart and period selector"
```

---

## Task 20: Frontend — Coach Pages

**Files:**
- Create: `src/frontend/src/components/coach/student-card.tsx`
- Create: `src/frontend/src/components/coach/diagnostics-list.tsx`
- Create: `src/frontend/src/app/(app)/dashboard/page.tsx`
- Create: `src/frontend/src/app/(app)/students/[id]/page.tsx`
- Create: `src/frontend/src/app/(app)/connections/page.tsx`

- [ ] **Step 1: Create student card**

```tsx
// src/frontend/src/components/coach/student-card.tsx
"use client"

import Link from "next/link"
import { Clock } from "lucide-react"
import type { Relationship } from "@/types"

export function StudentCard({ rel }: { rel: Relationship }) {
  return (
    <Link href={`/students/${rel.skater_id}`} className="block">
      <div className="rounded-2xl border border-border p-4 hover:bg-accent/30 transition-colors">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-full bg-muted flex items-center justify-center text-sm font-medium">
            {(rel.skater_name ?? "?")[0].toUpperCase()}
          </div>
          <div>
            <p className="font-medium text-sm">{rel.skater_name ?? "Ученик"}</p>
            <p className="text-xs text-muted-foreground flex items-center gap-1">
              <Clock className="h-3 w-3" />
              {new Date(rel.created_at).toLocaleDateString("ru-RU")}
            </p>
          </div>
        </div>
      </div>
    </Link>
  )
}
```

- [ ] **Step 2: Create diagnostics list**

```tsx
// src/frontend/src/components/coach/diagnostics-list.tsx
"use client"

import { AlertTriangle, Info } from "lucide-react"
import type { DiagnosticsFinding } from "@/types"

export function DiagnosticsList({ findings }: { findings: DiagnosticsFinding[] }) {
  if (!findings.length) {
    return <p className="text-sm text-muted-foreground">Проблем не обнаружено</p>
  }

  return (
    <div className="space-y-2">
      {findings.map((f, i) => (
        <div
          key={i}
          className={`rounded-xl border p-3 ${
            f.severity === "warning" ? "border-amber-300 bg-amber-50 dark:bg-amber-950/20" : "border-border bg-muted/30"
          }`}
        >
          <div className="flex items-start gap-2">
            {f.severity === "warning" ? (
              <AlertTriangle className="h-4 w-4 text-amber-500 mt-0.5 shrink-0" />
            ) : (
              <Info className="h-4 w-4 text-blue-500 mt-0.5 shrink-0" />
            )}
            <div>
              <p className="text-sm font-medium">{f.message}</p>
              <p className="text-xs text-muted-foreground mt-0.5">{f.detail}</p>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}
```

- [ ] **Step 3: Create dashboard page**

```tsx
// src/frontend/src/app/(app)/dashboard/page.tsx
"use client"

import { useRelationships } from "@/lib/api/relationships"
import { StudentCard } from "@/components/coach/student-card"

export default function DashboardPage() {
  const { data, isLoading } = useRelationships()

  const students = (data?.relationships ?? []).filter(
    (r) => r.status === "active",
  )

  if (isLoading) return <div className="py-20 text-center text-muted-foreground">Загрузка...</div>

  if (!students.length) {
    return (
      <div className="text-center py-20">
        <p className="text-muted-foreground">Нет учеников</p>
        <p className="text-sm text-muted-foreground mt-1">Пригласите первого ученика</p>
      </div>
    )
  }

  return (
    <div className="max-w-lg mx-auto space-y-3">
      <h1 className="text-lg font-semibold">Ученики</h1>
      {students.map((rel) => (
        <StudentCard key={rel.id} rel={rel} />
      ))}
    </div>
  )
}
```

- [ ] **Step 4: Create student profile page**

```tsx
// src/frontend/src/app/(app)/students/[id]/page.tsx
"use client"

import { useState } from "react"
import { useParams } from "next/navigation"
import Link from "next/link"
import { useDiagnostics, useTrend } from "@/lib/api/metrics"
import { DiagnosticsList } from "@/components/coach/diagnostics-list"
import { TrendChart } from "@/components/progress/trend-chart"
import { PeriodSelector } from "@/components/progress/period-selector"

const ELEMENTS = [
  { id: "three_turn", label: "Тройка" }, { id: "waltz_jump", label: "Вальсовый" },
  { id: "toe_loop", label: "Перекидной" }, { id: "flip", label: "Флип" },
  { id: "salchow", label: "Сальхов" }, { id: "loop", label: "Петля" },
  { id: "lutz", label: "Лютц" }, { id: "axel", label: "Аксель" },
]

export default function StudentProfilePage() {
  const { id } = useParams<{ id: string }>()
  const [tab, setTab] = useState<"progress" | "diagnostics">("progress")
  const [element, setElement] = useState("waltz_jump")
  const [metric, setMetric] = useState("max_height")
  const [period, setPeriod] = useState("30d")

  const { data: trend } = useTrend(id, element, metric, period)
  const { data: diag } = useDiagnostics(id)

  return (
    <div className="max-w-2xl mx-auto space-y-4">
      <div className="flex gap-2">
        <Link href="/dashboard" className="text-sm text-muted-foreground hover:underline">&larr; Назад</Link>
      </div>

      <div className="flex gap-1 rounded-lg bg-muted p-1">
        <button onClick={() => setTab("progress")} className={`flex-1 rounded-md px-3 py-2 text-sm font-medium ${tab === "progress" ? "bg-background shadow-sm" : ""}`}>
          Прогресс
        </button>
        <button onClick={() => setTab("diagnostics")} className={`flex-1 rounded-md px-3 py-2 text-sm font-medium ${tab === "diagnostics" ? "bg-background shadow-sm" : ""}`}>
          Диагностика
        </button>
      </div>

      {tab === "progress" && (
        <div className="space-y-4">
          <div className="grid grid-cols-4 gap-2">
            {ELEMENTS.map((el) => (
              <button key={el.id} onClick={() => setElement(el.id)} className={`rounded-xl border p-2 text-center text-xs ${element === el.id ? "border-primary bg-primary/10" : "border-border"}`}>
                {el.label}
              </button>
            ))}
          </div>
          <select value={metric} onChange={(e) => setMetric(e.target.value)} className="w-full rounded-xl border border-border bg-background px-3 py-2 text-sm">
            <option value="max_height">Высота прыжка</option>
            <option value="airtime">Время полёта</option>
            <option value="landing_knee_stability">Стабильность приземления</option>
            <option value="rotation_speed">Скорость вращения</option>
          </select>
          <PeriodSelector value={period} onChange={setPeriod} />
          {trend && <TrendChart data={trend} />}
        </div>
      )}

      {tab === "diagnostics" && diag && <DiagnosticsList findings={diag.findings} />}
    </div>
  )
}
```

- [ ] **Step 5: Create connections page**

```tsx
// src/frontend/src/app/(app)/connections/page.tsx
"use client"

import { useState } from "react"
import { toast } from "sonner"
import { useInvite, useRelationships, usePendingInvites, useAcceptInvite, useEndRelationship } from "@/lib/api/relationships"

export default function ConnectionsPage() {
  const { data: rels } = useRelationships()
  const { data: pending } = usePendingInvites()
  const invite = useInvite()
  const acceptInvite = useAcceptInvite()
  const endRel = useEndRelationship()

  const [email, setEmail] = useState("")

  const handleInvite = async () => {
    if (!email) return
    try {
      await invite.mutateAsync({ skater_email: email })
      toast.success("Приглашение отправлено")
      setEmail("")
    } catch {
      toast.error("Ошибка отправки")
    }
  }

  const activeRels = (rels?.relationships ?? []).filter((r) => r.status === "active")

  return (
    <div className="max-w-lg mx-auto space-y-6">
      <h1 className="text-lg font-semibold">Связи</h1>

      <div className="space-y-2">
        <p className="text-sm font-medium">Пригласить ученика</p>
        <div className="flex gap-2">
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="email@example.com"
            className="flex-1 rounded-xl border border-border bg-background px-3 py-2 text-sm"
          />
          <button onClick={handleInvite} className="rounded-xl bg-primary text-primary-foreground px-4 py-2 text-sm">
            Пригласить
          </button>
        </div>
      </div>

      {pending && pending.relationships.length > 0 && (
        <div className="space-y-2">
          <p className="text-sm font-medium">Входящие приглашения</p>
          {pending.relationships.map((r) => (
            <div key={r.id} className="flex items-center justify-between rounded-xl border border-border p-3">
              <span className="text-sm">{r.coach_name ?? r.coach_id}</span>
              <button onClick={() => acceptInvite.mutateAsync(r.id)} className="rounded-lg bg-primary px-3 py-1 text-xs text-primary-foreground">
                Принять
              </button>
            </div>
          ))}
        </div>
      )}

      {activeRels.length > 0 && (
        <div className="space-y-2">
          <p className="text-sm font-medium">Активные связи</p>
          {activeRels.map((r) => (
            <div key={r.id} className="flex items-center justify-between rounded-xl border border-border p-3">
              <span className="text-sm">{r.skater_name ?? r.skater_id}</span>
              <button onClick={() => endRel.mutateAsync(r.id)} className="text-xs text-muted-foreground hover:text-red-500">
                Завершить
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 6: Commit**

```bash
git add src/frontend/src/components/coach/ src/frontend/src/app/\(app\)/dashboard/ src/frontend/src/app/\(app\)/students/ src/frontend/src/app/\(app\)/connections/
git commit -m "feat(frontend): add coach dashboard, student profile, diagnostics, connections pages"
```

---

## Task 21: Root Layout Redirect

**Files:**
- Modify: `src/frontend/src/app/layout.tsx`

- [ ] **Step 1: Update root layout to redirect based on auth state**

The root `/` should redirect: logged out → `/login`, logged in with students → `/dashboard`, otherwise → `/feed`. This is handled client-side in the root `page.tsx`, not in `layout.tsx` (layout.tsx wraps all pages including login).

Create `src/frontend/src/app/page.tsx`:

```tsx
// src/frontend/src/app/page.tsx
"use client"

import { useAuth } from "@/components/auth-provider"
import { useRouter } from "next/navigation"
import { useEffect } from "react"
import { useRelationships } from "@/lib/api/relationships"

export default function HomePage() {
  const { user, isLoading } = useAuth()
  const router = useRouter()
  const { data: rels } = useRelationships()

  useEffect(() => {
    if (isLoading) return
    if (!user) {
      router.replace("/login")
      return
    }
    const hasStudents = (rels?.relationships ?? []).some((r) => r.status === "active")
    router.replace(hasStudents ? "/dashboard" : "/feed")
  }, [user, isLoading, rels, router])

  return null
}
```

- [ ] **Step 2: Commit**

```bash
git add src/frontend/src/app/page.tsx
git commit -m "feat(frontend): add root page redirect logic"
```

---

## Task 22: Integration Test & Verification

**Files:**
- No new files — manual verification

- [ ] **Step 1: Run all backend tests**

Run: `uv run pytest tests/backend/ -v`
Expected: All tests pass (metrics_registry, pr_tracker, diagnostics)

- [ ] **Step 2: Run Alembic migration**

Run: `uv run alembic upgrade head`
Expected: No new migrations needed (already applied)

- [ ] **Step 3: Start backend and verify endpoints**

Run: `uv run uvicorn src.backend.main:app --reload`
Verify:
- `GET /api/v1/metrics/registry` returns 12 metrics
- `POST /api/v1/auth/register` creates a user
- `POST /api/v1/sessions` creates a session
- `GET /api/v1/sessions` returns the session

- [ ] **Step 4: Start frontend and verify pages**

Run: `cd src/frontend && bun dev`
Verify:
- `/login` renders
- `/feed` renders after login
- `/upload` shows element picker
- `/progress` shows chart selectors
- `/dashboard` shows student list (empty initially)
- `/connections` shows invite form

- [ ] **Step 5: Commit any fixes found during verification**

```bash
git add -A
git commit -m "fix: integration test fixes"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] Role system (no fixed roles) — Task 3 (relationship CRUD), Task 9 (routes)
- [x] Session persistence — Task 2 (ORM), Task 6 (schemas), Task 7 (routes), Task 11 (saver)
- [x] SessionMetric with PR tracking — Task 2 (ORM), Task 4 (PR tracker)
- [x] Metric registry — Task 1
- [x] Overall score — Task 11 (session_saver)
- [x] PR direction — Task 1 (registry), Task 4 (pr_tracker)
- [x] Sessions API — Task 7
- [x] Metrics API (trend, PRs, diagnostics, registry) — Task 8
- [x] Relationships API — Task 9
- [x] Chunked upload — Task 10
- [x] Pipeline integration — Task 12
- [x] Body params in pipeline — Task 12
- [x] Activity feed — Task 16
- [x] Upload flow (simplified + camera + chunked) — Task 17
- [x] Session detail — Task 18
- [x] Progress charts — Task 19
- [x] Coach dashboard — Task 20
- [x] Student profile — Task 20
- [x] Connections page — Task 20
- [x] Responsive layout — Task 15
- [x] PATCH sessions — Task 7
- [x] Diagnostics engine — Task 5

**Placeholder scan:**
- No TBDs, TODOs, or "implement later" found
- All code steps have actual implementations
- All test steps have actual test code

**Type consistency:**
- `SessionMetric.metric_name` used consistently across CRUD, routes, services
- `Relationship.status` values ("invited", "active", "ended") consistent
- `METRIC_REGISTRY` direction values ("higher", "lower") consistent with PR tracker
- Frontend types match backend schemas
