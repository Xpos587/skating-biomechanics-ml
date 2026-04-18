# Flexible Connections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace hardcoded `Relationship(coach_id, skater_id)` with flexible `Connection(from_user_id, to_user_id, connection_type)` supporting coaching and choreography roles.

**Architecture:** User model stays clean (no role field). Role is determined by `connection_type` on the `Connection` model. `from_user_id` is the guiding party (coach/choreographer), `to_user_id` is the learning party (student/skater). `initiated_by` tracks who pressed "invite" — can be either party. One user can be coach in one connection and student in another. Sport dimension deferred.

**Tech Stack:** SQLAlchemy (async), Alembic, FastAPI, Pydantic, pytest + pytest-asyncio, React Query + Zod (frontend)

---

## File Map

### Backend — Create
| File | Responsibility |
|------|---------------|
| `backend/app/models/connection.py` | Connection ORM model with ConnectionType/ConnectionStatus enums |
| `backend/app/crud/connection.py` | CRUD: create, get, list, is_connected_as |
| `backend/app/routes/connections.py` | API: POST/GET /connections, accept, end, pending |
| `backend/alembic/versions/YYYY-MM-DD-HHMM_<hash>_relationships_to_connections.py` | Migration: rename table, add connection_type, rename columns |

### Backend — Modify
| File | Change |
|------|--------|
| `backend/app/models/__init__.py` | Replace `Relationship` import with `Connection` |
| `backend/app/schemas.py` | Replace Relationship schemas with Connection schemas |
| `backend/app/routes/sessions.py` | Replace `is_coach_for_student` → `is_connected_as` |
| `backend/app/routes/metrics.py` | Replace `is_coach_for_student` → `is_connected_as` |
| `backend/app/main.py` | Replace `relationships` router import with `connections` |

### Backend — Delete
| File | Reason |
|------|--------|
| `backend/app/models/relationship.py` | Replaced by connection.py |
| `backend/app/crud/relationship.py` | Replaced by connection.py |
| `backend/app/routes/relationships.py` | Replaced by connections.py |

### Frontend — Modify
| File | Change |
|------|--------|
| `frontend/src/types/index.ts` | Replace `Relationship` interface with `Connection` |
| `frontend/src/lib/api/relationships.ts` | Update Zod schemas + API paths to `/connections` |
| `frontend/src/app/(app)/connections/page.tsx` | Use `from_user_id`/`to_user_id` + `connection_type` |
| `frontend/src/app/(app)/dashboard/page.tsx` | Filter by `connection_type === "coaching"` |
| `frontend/src/components/coach/student-card.tsx` | Use `to_user_id` instead of `skater_id` |
| `frontend/src/components/app-nav.tsx` | Update queryKey + schema |
| `frontend/src/components/layout/bottom-dock.tsx` | Update queryKey + schema |

### Backend — Tests
| File | Responsibility |
|------|---------------|
| `backend/tests/models/test_connection.py` | ORM model tests |
| `backend/tests/crud/test_connection.py` | CRUD function tests |

---

## Task 1: Connection ORM Model

**Files:**

- Create: `backend/app/models/connection.py`
- Modify: `backend/app/models/__init__.py`
- Test: `backend/tests/models/test_connection.py`

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/models/test_connection.py
"""Tests for Connection ORM model."""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.connection import Connection, ConnectionStatus, ConnectionType


@pytest.mark.asyncio
async def test_create_connection(db_session: AsyncSession):
    """Connection created with defaults."""
    conn = Connection(
        from_user_id="coach-1",
        to_user_id="student-1",
        connection_type=ConnectionType.COACHING,
        initiated_by="coach-1",
    )
    db_session.add(conn)
    await db_session.flush()
    await db_session.refresh(conn)

    assert conn.id is not None
    assert conn.from_user_id == "coach-1"
    assert conn.to_user_id == "student-1"
    assert conn.connection_type == ConnectionType.COACHING
    assert conn.status == ConnectionStatus.INVITED
    assert conn.initiated_by == "coach-1"
    assert conn.ended_at is None
    assert conn.created_at is not None


@pytest.mark.asyncio
async def test_connection_type_enum_values():
    """ConnectionType has expected members."""
    assert ConnectionType.COACHING.value == "coaching"
    assert ConnectionType.CHOREOGRAPHY.value == "choreography"


@pytest.mark.asyncio
async def test_connection_status_enum_values():
    """ConnectionStatus has expected members."""
    assert ConnectionStatus.INVITED.value == "invited"
    assert ConnectionStatus.ACTIVE.value == "active"
    assert ConnectionStatus.ENDED.value == "ended"


@pytest.mark.asyncio
async def test_connection_choreography_type(db_session: AsyncSession):
    """Connection works with choreography type."""
    conn = Connection(
        from_user_id="choreographer-1",
        to_user_id="skater-1",
        connection_type=ConnectionType.CHOREOGRAPHY,
        initiated_by="skater-1",
    )
    db_session.add(conn)
    await db_session.flush()
    await db_session.refresh(conn)

    assert conn.connection_type == ConnectionType.CHOREOGRAPHY


@pytest.mark.asyncio
async def test_connection_read_back(db_session: AsyncSession):
    """Connection persists and reads back correctly."""
    conn = Connection(
        from_user_id="u1",
        to_user_id="u2",
        connection_type=ConnectionType.COACHING,
        initiated_by="u1",
    )
    db_session.add(conn)
    await db_session.flush()

    result = await db_session.execute(select(Connection).where(Connection.from_user_id == "u1"))
    fetched = result.scalar_one()
    assert fetched.to_user_id == "u2"
    assert fetched.connection_type == ConnectionType.COACHING
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/michael/Github/skating-biomechanics-ml && uv run pytest backend/tests/models/test_connection.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.models.connection'`

- [ ] **Step 3: Create the Connection model**

```python
# backend/app/models/connection.py
"""Connection ORM model for flexible user-to-user relationships."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime  # noqa: TC003

from sqlalchemy import DateTime, Enum, ForeignKey, String, text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class ConnectionType(str, enum.Enum):
    COACHING = "coaching"
    CHOREOGRAPHY = "choreography"


class ConnectionStatus(str, enum.Enum):
    INVITED = "invited"
    ACTIVE = "active"
    ENDED = "ended"


class Connection(TimestampMixin, Base):
    """User-to-user connection with flexible role semantics.

    from_user_id = guiding party (coach, choreographer)
    to_user_id = learning party (student, skater)
    initiated_by = who created the connection (either party)
    """

    __tablename__ = "connections"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    from_user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
    )
    to_user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
    )
    connection_type: Mapped[ConnectionType] = mapped_column(
        Enum(ConnectionType, name="connection_type", length=20),
        default=ConnectionType.COACHING,
    )
    status: Mapped[ConnectionStatus] = mapped_column(
        Enum(ConnectionStatus, name="connection_status", length=20),
        default=ConnectionStatus.INVITED,
    )
    initiated_by: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="SET NULL"),
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        # Partial unique index: one active connection per user pair + type
        text(
            "CREATE UNIQUE INDEX uq_connection_active ON connections "
            "(from_user_id, to_user_id, connection_type) "
            "WHERE status != 'ended'"
        ),
    )
```

- [ ] **Step 4: Update models/__init__.py**

Replace the `Relationship` import with `Connection`:

```python
# backend/app/models/__init__.py
"""SQLAlchemy ORM models."""

from app.models.base import Base
from app.models.connection import Connection
from app.models.choreography import ChoreographyProgram, MusicAnalysis
from app.models.refresh_token import RefreshToken
from app.models.session import Session, SessionMetric
from app.models.user import User

__all__ = [
    "Base",
    "ChoreographyProgram",
    "Connection",
    "MusicAnalysis",
    "RefreshToken",
    "Session",
    "SessionMetric",
    "User",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/michael/Github/skating-biomechanics-ml && uv run pytest backend/tests/models/test_connection.py -v`
Expected: 5 passed

- [ ] **Step 6: Run all existing model tests to check nothing broke**

Run: `cd /home/michael/Github/skating-biomechanics-ml && uv run pytest backend/tests/test_models.py -v`
Expected: all passed (no Relationship import used in these tests)

- [ ] **Step 7: Commit**

```bash
git add backend/app/models/connection.py backend/app/models/__init__.py backend/tests/models/test_connection.py
git commit -m "feat(models): add Connection model with ConnectionType enum"
```

---

## Task 2: Connection CRUD

**Files:**

- Create: `backend/app/crud/connection.py`
- Test: `backend/tests/crud/test_connection.py`

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/crud/test_connection.py
"""Tests for Connection CRUD operations."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.crud.connection import (
    create,
    get_active,
    get_by_id,
    is_connected_as,
    list_for_user,
    list_pending_for_user,
)
from app.models.connection import Connection, ConnectionStatus, ConnectionType
from app.models.user import User


@pytest.fixture
async def coach(db_session: AsyncSession) -> User:
    u = User(email="coach@test.com", hashed_password="h")
    db_session.add(u)
    await db_session.flush()
    return u


@pytest.fixture
async def student(db_session: AsyncSession) -> User:
    u = User(email="student@test.com", hashed_password="h")
    db_session.add(u)
    await db_session.flush()
    return u


@pytest.mark.asyncio
async def test_create_connection(db_session: AsyncSession, coach: User, student: User):
    conn = await create(
        db_session,
        from_user_id=coach.id,
        to_user_id=student.id,
        connection_type=ConnectionType.COACHING,
        initiated_by=coach.id,
    )
    assert conn.from_user_id == coach.id
    assert conn.to_user_id == student.id
    assert conn.connection_type == ConnectionType.COACHING
    assert conn.status == ConnectionStatus.INVITED


@pytest.mark.asyncio
async def test_get_by_id(db_session: AsyncSession, coach: User, student: User):
    conn = await create(
        db_session,
        from_user_id=coach.id,
        to_user_id=student.id,
        connection_type=ConnectionType.COACHING,
        initiated_by=coach.id,
    )
    fetched = await get_by_id(db_session, conn.id)
    assert fetched is not None
    assert fetched.from_user_id == coach.id


@pytest.mark.asyncio
async def test_get_by_id_not_found(db_session: AsyncSession):
    fetched = await get_by_id(db_session, "nonexistent")
    assert fetched is None


@pytest.mark.asyncio
async def test_get_active(db_session: AsyncSession, coach: User, student: User):
    await create(
        db_session,
        from_user_id=coach.id,
        to_user_id=student.id,
        connection_type=ConnectionType.COACHING,
        initiated_by=coach.id,
    )
    active = await get_active(
        db_session,
        from_user_id=coach.id,
        to_user_id=student.id,
        connection_type=ConnectionType.COACHING,
    )
    assert active is not None


@pytest.mark.asyncio
async def test_get_active_ended_returns_none(db_session: AsyncSession, coach: User, student: User):
    conn = await create(
        db_session,
        from_user_id=coach.id,
        to_user_id=student.id,
        connection_type=ConnectionType.COACHING,
        initiated_by=coach.id,
    )
    conn.status = ConnectionStatus.ENDED
    db_session.add(conn)
    await db_session.flush()

    active = await get_active(
        db_session,
        from_user_id=coach.id,
        to_user_id=student.id,
        connection_type=ConnectionType.COACHING,
    )
    assert active is None


@pytest.mark.asyncio
async def test_list_for_user(db_session: AsyncSession, coach: User, student: User):
    await create(
        db_session,
        from_user_id=coach.id,
        to_user_id=student.id,
        connection_type=ConnectionType.COACHING,
        initiated_by=coach.id,
    )
    coach_conns = await list_for_user(db_session, coach.id)
    student_conns = await list_for_user(db_session, student.id)
    assert len(coach_conns) == 1
    assert len(student_conns) == 1


@pytest.mark.asyncio
async def test_list_pending_for_user(db_session: AsyncSession, coach: User, student: User):
    await create(
        db_session,
        from_user_id=coach.id,
        to_user_id=student.id,
        connection_type=ConnectionType.COACHING,
        initiated_by=coach.id,
    )
    pending = await list_pending_for_user(db_session, student.id)
    assert len(pending) == 1
    assert pending[0].status == ConnectionStatus.INVITED


@pytest.mark.asyncio
async def test_is_connected_as_true(db_session: AsyncSession, coach: User, student: User):
    await create(
        db_session,
        from_user_id=coach.id,
        to_user_id=student.id,
        connection_type=ConnectionType.COACHING,
        initiated_by=coach.id,
    )
    assert await is_connected_as(
        db_session,
        from_user_id=coach.id,
        to_user_id=student.id,
        connection_type=ConnectionType.COACHING,
    ) is True


@pytest.mark.asyncio
async def test_is_connected_as_false(db_session: AsyncSession, coach: User, student: User):
    assert await is_connected_as(
        db_session,
        from_user_id=coach.id,
        to_user_id=student.id,
        connection_type=ConnectionType.COACHING,
    ) is False


@pytest.mark.asyncio
async def test_is_connected_as_wrong_type(db_session: AsyncSession, coach: User, student: User):
    """coaching connection does not grant choreography access."""
    await create(
        db_session,
        from_user_id=coach.id,
        to_user_id=student.id,
        connection_type=ConnectionType.COACHING,
        initiated_by=coach.id,
    )
    assert await is_connected_as(
        db_session,
        from_user_id=coach.id,
        to_user_id=student.id,
        connection_type=ConnectionType.CHOREOGRAPHY,
    ) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/michael/Github/skating-biomechanics-ml && uv run pytest backend/tests/crud/test_connection.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.crud.connection'`

- [ ] **Step 3: Implement CRUD**

```python
# backend/app/crud/connection.py
"""Connection CRUD operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select

from app.models.connection import Connection, ConnectionStatus, ConnectionType

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def create(
    db: AsyncSession,
    *,
    from_user_id: str,
    to_user_id: str,
    connection_type: ConnectionType,
    initiated_by: str,
) -> Connection:
    conn = Connection(
        from_user_id=from_user_id,
        to_user_id=to_user_id,
        connection_type=connection_type,
        initiated_by=initiated_by,
    )
    db.add(conn)
    await db.flush()
    await db.refresh(conn)
    return conn


async def get_by_id(db: AsyncSession, conn_id: str) -> Connection | None:
    result = await db.execute(select(Connection).where(Connection.id == conn_id))
    return result.scalar_one_or_none()


async def get_active(
    db: AsyncSession,
    *,
    from_user_id: str,
    to_user_id: str,
    connection_type: ConnectionType,
) -> Connection | None:
    result = await db.execute(
        select(Connection).where(
            Connection.from_user_id == from_user_id,
            Connection.to_user_id == to_user_id,
            Connection.connection_type == connection_type,
            Connection.status != ConnectionStatus.ENDED,
        )
    )
    return result.scalar_one_or_none()


async def list_for_user(db: AsyncSession, user_id: str) -> list[Connection]:
    """List all connections where user is either party."""
    result = await db.execute(
        select(Connection)
        .where(
            (Connection.from_user_id == user_id) | (Connection.to_user_id == user_id),
        )
        .order_by(Connection.created_at.desc())
    )
    return list(result.scalars().all())


async def list_pending_for_user(db: AsyncSession, user_id: str) -> list[Connection]:
    """List pending invites received by user (as to_user)."""
    result = await db.execute(
        select(Connection)
        .where(
            Connection.to_user_id == user_id,
            Connection.status == ConnectionStatus.INVITED,
        )
        .order_by(Connection.created_at.desc())
    )
    return list(result.scalars().all())


async def is_connected_as(
    db: AsyncSession,
    *,
    from_user_id: str,
    to_user_id: str,
    connection_type: ConnectionType,
) -> bool:
    """Check if an active connection exists between two users of a specific type."""
    result = await db.execute(
        select(Connection).where(
            Connection.from_user_id == from_user_id,
            Connection.to_user_id == to_user_id,
            Connection.connection_type == connection_type,
            Connection.status == ConnectionStatus.ACTIVE,
        )
    )
    return result.scalar_one_or_none() is not None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/michael/Github/skating-biomechanics-ml && uv run pytest backend/tests/crud/test_connection.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add backend/app/crud/connection.py backend/tests/crud/test_connection.py
git commit -m "feat(crud): add Connection CRUD with is_connected_as"
```

---

## Task 3: Pydantic Schemas

**Files:**

- Modify: `backend/app/schemas.py`

- [ ] **Step 1: Replace relationship schemas with connection schemas**

In `backend/app/schemas.py`, replace the `# Relationships` section (lines ~306-338) with:

```python
# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------


class InviteRequest(BaseModel):
    to_user_email: str
    connection_type: str = Field(pattern=r"^(coaching|choreography)$")


class ConnectionResponse(BaseModel):
    id: str
    from_user_id: str
    to_user_id: str
    connection_type: str
    status: str
    initiated_by: str | None
    created_at: str
    ended_at: str | None
    from_user_name: str | None = None
    to_user_name: str | None = None

    model_config = {"from_attributes": True}

    @field_validator("created_at", "ended_at", mode="before")
    @classmethod
    def validate_datetime(cls, v):
        if isinstance(v, datetime):
            return v.isoformat()
        return str(v)


class ConnectionListResponse(BaseModel):
    connections: list[ConnectionResponse]
```

- [ ] **Step 2: Verify no import errors**

Run: `cd /home/michael/Github/skating-biomechanics-ml && uv run python -c "from app.schemas import InviteRequest, ConnectionResponse, ConnectionListResponse; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add backend/app/schemas.py
git commit -m "feat(schemas): replace Relationship schemas with Connection schemas"
```

---

## Task 4: Connection Routes

**Files:**

- Create: `backend/app/routes/connections.py`
- Modify: `backend/app/main.py`

- [ ] **Step 1: Create the connections router**

```python
# backend/app/routes/connections.py
"""Flexible connection API routes."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, status

from app.crud.connection import (
    create as create_conn,
)
from app.crud.connection import (
    get_active as get_active_conn,
)
from app.crud.connection import (
    get_by_id as get_conn_by_id,
)
from app.crud.connection import (
    list_for_user,
    list_pending_for_user,
)
from app.crud.user import get_by_email
from app.models.connection import ConnectionStatus, ConnectionType
from app.schemas import ConnectionListResponse, ConnectionResponse, InviteRequest

if TYPE_CHECKING:
    from app.auth.deps import CurrentUser, DbDep
    from app.models.connection import Connection


router = APIRouter(tags=["connections"])


def _conn_to_response(conn: Connection) -> ConnectionResponse:
    return ConnectionResponse.model_validate(conn)


@router.post(
    "/connections/invite",
    response_model=ConnectionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def invite(body: InviteRequest, user: CurrentUser, db: DbDep):
    """User invites another user to a connection (coaching, choreography)."""
    to_user = await get_by_email(db, body.to_user_email)
    if not to_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    conn_type = ConnectionType(body.connection_type)

    existing = await get_active_conn(
        db,
        from_user_id=user.id,
        to_user_id=to_user.id,
        connection_type=conn_type,
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Connection already exists"
        )

    conn = await create_conn(
        db,
        from_user_id=user.id,
        to_user_id=to_user.id,
        connection_type=conn_type,
        initiated_by=user.id,
    )
    return _conn_to_response(conn)


@router.post("/connections/{conn_id}/accept", response_model=ConnectionResponse)
async def accept_invite(conn_id: str, user: CurrentUser, db: DbDep):
    """Invitee accepts a connection."""
    conn = await get_conn_by_id(db, conn_id)
    if not conn:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connection not found")
    if conn.to_user_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    if conn.status != ConnectionStatus.INVITED:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Not an active invite")

    conn.status = ConnectionStatus.ACTIVE
    db.add(conn)
    await db.flush()
    await db.refresh(conn)
    return _conn_to_response(conn)


@router.post("/connections/{conn_id}/end", response_model=ConnectionResponse)
async def end_connection(conn_id: str, user: CurrentUser, db: DbDep):
    """Either party ends the connection."""
    conn = await get_conn_by_id(db, conn_id)
    if not conn:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connection not found")
    if user.id not in (conn.from_user_id, conn.to_user_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")
    if conn.status == ConnectionStatus.ENDED:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Already ended")

    conn.status = ConnectionStatus.ENDED
    conn.ended_at = datetime.now(UTC)
    db.add(conn)
    await db.flush()
    await db.refresh(conn)
    return _conn_to_response(conn)


@router.get("/connections", response_model=ConnectionListResponse)
async def list_connections(user: CurrentUser, db: DbDep):
    """List all connections for the current user."""
    conns = await list_for_user(db, user.id)
    return ConnectionListResponse(connections=[_conn_to_response(c) for c in conns])


@router.get("/connections/pending", response_model=ConnectionListResponse)
async def list_pending(user: CurrentUser, db: DbDep):
    """List pending invites received by the current user."""
    conns = await list_pending_for_user(db, user.id)
    return ConnectionListResponse(connections=[_conn_to_response(c) for c in conns])
```

- [ ] **Step 2: Wire up the router in main.py**

In `backend/app/main.py`, replace:

```python
from app.routes import (
    auth,
    choreography,
    detect,
    metrics,
    misc,
    models,
    process,
    relationships,
    sessions,
    uploads,
    users,
)
```

with:

```python
from app.routes import (
    auth,
    choreography,
    connections,
    detect,
    metrics,
    misc,
    models,
    process,
    sessions,
    uploads,
    users,
)
```

And replace:

```python
api_v1.include_router(relationships.router)
```

with:

```python
api_v1.include_router(connections.router)
```

- [ ] **Step 3: Verify app starts**

Run: `cd /home/michael/Github/skating-biomechanics-ml && uv run python -c "from app.main import app; print('Routes:', [r.path for r in app.routes])"`
Expected: `/connections` and `/connections/pending` in the output, no `/relationships`

- [ ] **Step 4: Commit**

```bash
git add backend/app/routes/connections.py backend/app/main.py
git commit -m "feat(routes): add /connections API with invite/accept/end"
```

---

## Task 5: Update Sessions & Metrics Routes

**Files:**

- Modify: `backend/app/routes/sessions.py`
- Modify: `backend/app/routes/metrics.py`

- [ ] **Step 1: Update sessions.py**

Replace the import:

```python
from app.crud.relationship import is_coach_for_student
```

with:

```python
from app.crud.connection import is_connected_as
from app.models.connection import ConnectionType
```

Replace `is_coach_for_student(db, coach_id=user.id, skater_id=user_id)` (line 75) with:

```python
is_connected_as(db, from_user_id=user.id, to_user_id=user_id, connection_type=ConnectionType.COACHING)
```

Replace `is_coach_for_student(db, coach_id=user.id, skater_id=session.user_id)` (line 99-100) with:

```python
is_connected_as(db, from_user_id=user.id, to_user_id=session.user_id, connection_type=ConnectionType.COACHING)
```

- [ ] **Step 2: Update metrics.py**

Replace the import:

```python
from app.crud.relationship import is_coach_for_student
```

with:

```python
from app.crud.connection import is_connected_as
from app.models.connection import ConnectionType
```

Replace all 3 occurrences of:

```python
is_coach_for_student(db, coach_id=user.id, skater_id=user_id)
```

with:

```python
is_connected_as(db, from_user_id=user.id, to_user_id=user_id, connection_type=ConnectionType.COACHING)
```

- [ ] **Step 3: Verify no import errors**

Run: `cd /home/michael/Github/skating-biomechanics-ml && uv run python -c "from app.main import app; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Run all backend tests**

Run: `cd /home/michael/Github/skating-biomechanics-ml && uv run pytest backend/tests/ -v --timeout=30`
Expected: all passed (no relationship references remain in test code)

- [ ] **Step 5: Commit**

```bash
git add backend/app/routes/sessions.py backend/app/routes/metrics.py
git commit -m "refactor(routes): use is_connected_as instead of is_coach_for_student"
```

---

## Task 6: Alembic Migration

**Files:**

- Create: `backend/alembic/versions/YYYY_MM_DD_HHMM_<hash>_relationships_to_connections.py`

- [ ] **Step 1: Generate migration stub**

Run: `cd /home/michael/Github/skating-biomechanics-ml/backend && uv run alembic revision -m "relationships_to_connections"`
Expected: New file in `alembic/versions/`

- [ ] **Step 2: Write the migration**

The migration needs to:
1. Rename table `relationships` → `connections`
2. Rename column `coach_id` → `from_user_id`
3. Rename column `skater_id` → `to_user_id`
4. Add column `connection_type` (VARCHAR(20), default `'coaching'`)
5. Change `status` to use enum (or keep as VARCHAR for SQLite compat — keep VARCHAR)
6. Drop old partial unique index `uq_coach_skater_active`
7. Create new partial unique index `uq_connection_active`
8. Rename `initiated_by` FK stays the same

```python
"""relationships_to_connections

Revision ID: <auto>
Revises: 1541cafaf37d
Create Date: 2026-04-17

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "<auto>"
down_revision: str | Sequence[str] | None = "1541cafaf37d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Drop old partial unique index
    op.drop_index("uq_coach_skater_active", table_name="relationships")

    # 2. Rename table
    op.rename_table("relationships", "connections")

    # 3. Rename columns
    op.alter_column("connections", "coach_id", new_column_name="from_user_id")
    op.alter_column("connections", "skater_id", new_column_name="to_user_id")

    # 4. Add connection_type column (default 'coaching' for existing rows)
    op.add_column(
        "connections",
        sa.Column("connection_type", sa.String(length=20), nullable=False, server_default="coaching"),
    )

    # 5. Create new partial unique index
    op.execute(
        "CREATE UNIQUE INDEX uq_connection_active ON connections "
        "(from_user_id, to_user_id, connection_type) "
        "WHERE status != 'ended'"
    )


def downgrade() -> None:
    # 1. Drop new index
    op.execute("DROP INDEX IF EXISTS uq_connection_active")

    # 2. Remove connection_type column
    op.drop_column("connections", "connection_type")

    # 3. Rename columns back
    op.alter_column("connections", "from_user_id", new_column_name="coach_id")
    op.alter_column("connections", "to_user_id", new_column_name="skater_id")

    # 4. Rename table back
    op.rename_table("connections", "relationships")

    # 5. Recreate old partial unique index
    op.create_index(
        "uq_coach_skater_active",
        "relationships",
        ["coach_id", "skater_id"],
        unique=True,
        postgresql_where=sa.text("status != 'ended'"),
    )
```

- [ ] **Step 3: Verify migration applies cleanly (SQLite)**

Run: `cd /home/michael/Github/skating-biomechanics-ml/backend && uv run alembic upgrade head`
Expected: No errors

- [ ] **Step 4: Verify migration downgrades cleanly**

Run: `cd /home/michael/Github/skating-biomechanics-ml/backend && uv run alembic downgrade -1`
Expected: No errors

- [ ] **Step 5: Re-apply**

Run: `cd /home/michael/Github/skating-biomechanics-ml/backend && uv run alembic upgrade head`

- [ ] **Step 6: Commit**

```bash
git add backend/alembic/versions/*relationships_to_connections*
git commit -m "feat(db): migrate relationships table to connections with connection_type"
```

---

## Task 7: Delete Old Relationship Files

**Files:**

- Delete: `backend/app/models/relationship.py`
- Delete: `backend/app/crud/relationship.py`
- Delete: `backend/app/routes/relationships.py`

- [ ] **Step 1: Verify nothing imports the old modules**

Run: `cd /home/michael/Github/skating-biomechanics-ml && grep -r "from app.models.relationship" backend/ && grep -r "from app.crud.relationship" backend/ && grep -r "from app.routes.relationships" backend/`
Expected: No output (all references already replaced in Tasks 4-5)

- [ ] **Step 2: Delete old files**

```bash
rm backend/app/models/relationship.py
rm backend/app/crud/relationship.py
rm backend/app/routes/relationships.py
```

- [ ] **Step 3: Run all backend tests**

Run: `cd /home/michael/Github/skating-biomechanics-ml && uv run pytest backend/tests/ -v --timeout=30`
Expected: all passed

- [ ] **Step 4: Commit**

```bash
git add -u backend/app/models/relationship.py backend/app/crud/relationship.py backend/app/routes/relationships.py
git commit -m "refactor: remove old Relationship model, crud, and routes"
```

---

## Task 8: Frontend TypeScript Types

**Files:**

- Modify: `frontend/src/types/index.ts`

- [ ] **Step 1: Replace Relationship types with Connection types**

In `frontend/src/types/index.ts`, replace the `// Relationships` section (lines ~152-170) with:

```typescript
// ---------------------------------------------------------------------------
// Connections
// ---------------------------------------------------------------------------

export type ConnectionType = "coaching" | "choreography"

export interface Connection {
  id: string
  from_user_id: string
  to_user_id: string
  connection_type: ConnectionType
  status: "invited" | "active" | "ended"
  initiated_by: string | null
  created_at: string
  ended_at: string | null
  from_user_name: string | null
  to_user_name: string | null
}

export interface ConnectionListResponse {
  connections: Connection[]
}
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd /home/michael/Github/skating-biomechanics-ml/frontend && bunx tsc --noEmit`
Expected: errors in files that still use old `Relationship` type (expected — fixed in Tasks 9-11)

- [ ] **Step 3: Commit**

```bash
git add frontend/src/types/index.ts
git commit -m "feat(frontend): replace Relationship type with Connection type"
```

---

## Task 9: Frontend API Client

**Files:**

- Modify: `frontend/src/lib/api/relationships.ts` (rename to `connections.ts`)

- [ ] **Step 1: Rewrite the API client**

Create `frontend/src/lib/api/connections.ts`:

```typescript
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { z } from "zod"
import { apiFetch, apiPost } from "@/lib/api-client"

const ConnectionSchema = z.object({
  id: z.string(),
  from_user_id: z.string(),
  to_user_id: z.string(),
  connection_type: z.enum(["coaching", "choreography"]),
  status: z.enum(["invited", "active", "ended"]),
  initiated_by: z.string().nullable(),
  created_at: z.string(),
  ended_at: z.string().nullable(),
  from_user_name: z.string().nullable(),
  to_user_name: z.string().nullable(),
})

const ConnectionListSchema = z.object({ connections: z.array(ConnectionSchema) })

export function useConnections() {
  return useQuery({
    queryKey: ["connections"],
    queryFn: () => apiFetch("/connections", ConnectionListSchema),
  })
}

export function usePendingConnections() {
  return useQuery({
    queryKey: ["connections", "pending"],
    queryFn: () => apiFetch("/connections/pending", ConnectionListSchema),
  })
}

export function useInviteConnection() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: { to_user_email: string; connection_type: "coaching" | "choreography" }) =>
      apiPost("/connections/invite", ConnectionSchema, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["connections"] }),
  })
}

export function useAcceptConnection() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (connId: string) =>
      apiPost(`/connections/${connId}/accept`, ConnectionSchema, {}),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["connections"] }),
  })
}

export function useEndConnection() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (connId: string) =>
      apiPost(`/connections/${connId}/end`, ConnectionSchema, {}),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["connections"] }),
  })
}
```

- [ ] **Step 2: Delete old file**

```bash
rm frontend/src/lib/api/relationships.ts
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/lib/api/connections.ts && git rm frontend/src/lib/api/relationships.ts
git commit -m "feat(frontend): rewrite relationships API as connections API"
```

---

## Task 10: Frontend Pages & Components

**Files:**

- Modify: `frontend/src/app/(app)/connections/page.tsx`
- Modify: `frontend/src/app/(app)/dashboard/page.tsx`
- Modify: `frontend/src/components/coach/student-card.tsx`
- Modify: `frontend/src/components/app-nav.tsx`
- Modify: `frontend/src/components/layout/bottom-dock.tsx`

- [ ] **Step 1: Update connections page**

In `frontend/src/app/(app)/connections/page.tsx`, replace:

```typescript
import {
  useAcceptInvite,
  useEndRelationship,
  useInvite,
  usePendingInvites,
  useRelationships,
} from "@/lib/api/relationships"
```

with:

```typescript
import {
  useAcceptConnection,
  useEndConnection,
  useInviteConnection,
  usePendingConnections,
  useConnections,
} from "@/lib/api/connections"
```

Replace hook calls in the component:

```typescript
const { data: conns } = useConnections()
const { data: pending } = usePendingConnections()
const invite = useInviteConnection()
const acceptConn = useAcceptConnection()
const endConn = useEndConnection()
```

Replace `invite.mutateAsync({ skater_email: email })` with:

```typescript
invite.mutateAsync({ to_user_email: email, connection_type: "coaching" })
```

Replace `rels?.relationships` with `conns?.connections`, `pending?.relationships` with `pending?.connections`.

Replace `r.coach_name ?? r.coach_id` with `r.from_user_name ?? r.from_user_id` (pending invites show the inviter's name).

Replace `r.skater_name ?? r.skater_id` with `r.to_user_name ?? r.to_user_id` (active connections show the other party's name).

Replace `acceptInvite.mutateAsync(r.id)` with `acceptConn.mutateAsync(r.id)`.

Replace `endRel.mutateAsync(r.id)` with `endConn.mutateAsync(r.id)`.

- [ ] **Step 2: Update dashboard page**

In `frontend/src/app/(app)/dashboard/page.tsx`, replace:

```typescript
import { useRelationships } from "@/lib/api/relationships"
```

with:

```typescript
import { useConnections } from "@/lib/api/connections"
```

Replace `useRelationships()` with `useConnections()`.

Replace `(data?.relationships ?? []).filter(r => r.status === "active")` with:

```typescript
(data?.connections ?? []).filter(r => r.status === "active" && r.connection_type === "coaching")
```

- [ ] **Step 3: Update student-card component**

In `frontend/src/components/coach/student-card.tsx`, replace:

```typescript
import type { Relationship } from "@/types"
```

with:

```typescript
import type { Connection } from "@/types"
```

Replace `rel: Relationship` with `conn: Connection`.

Replace `rel.skater_id` with `conn.to_user_id`.

Replace `rel.skater_name ?? rel.skater_id` with `conn.to_user_name ?? conn.to_user_id`.

- [ ] **Step 4: Update app-nav.tsx**

Replace queryKey `"relationships"` with `"connections"`, and update the Zod schema field name:

```typescript
const ConnectionListSchema = z.object({
  connections: z.array(z.object({ status: z.string(), connection_type: z.string() })),
})

const { data: connsData } = useQuery({
  queryKey: ["connections"],
  queryFn: () => apiFetch("/connections", ConnectionListSchema),
})
const hasStudents = (connsData?.connections ?? []).some(
  r => r.status === "active" && r.connection_type === "coaching",
)
```

- [ ] **Step 5: Update bottom-dock.tsx**

Same pattern as app-nav.tsx — replace `"relationships"` queryKey with `"connections"`, update schema field names:

```typescript
const ConnectionListSchema = z.object({
  connections: z.array(z.object({ status: z.string(), connection_type: z.string() })),
})

const { data: connsData } = useQuery({
  queryKey: ["connections"],
  queryFn: () => apiFetch("/connections", ConnectionListSchema),
})
const hasStudents = (connsData?.connections ?? []).some(
  r => r.status === "active" && r.connection_type === "coaching",
)
```

- [ ] **Step 6: Verify TypeScript compiles**

Run: `cd /home/michael/Github/skating-biomechanics-ml/frontend && bunx tsc --noEmit`
Expected: 0 errors

- [ ] **Step 7: Run lint**

Run: `cd /home/michael/Github/skating-biomechanics-ml/frontend && bunx next lint`
Expected: No errors

- [ ] **Step 8: Commit**

```bash
git add frontend/src/app/ frontend/src/components/ frontend/src/lib/api/connections.ts
git rm frontend/src/lib/api/relationships.ts
git commit -m "feat(frontend): update all components to use Connection model"
```

---

## Task 11: Update CLAUDE.md Documentation

**Files:**

- Modify: `backend/CLAUDE.md`
- Modify: `frontend/CLAUDE.md`

- [ ] **Step 1: Update backend/CLAUDE.md**

Replace `relationships.py` with `connections.py` in the project structure and API routes table. Replace `is_coach_for_student()` with `is_connected_as()` in auth architecture description. Add connection_type to the API route descriptions.

- [ ] **Step 2: Update frontend/CLAUDE.md**

Replace `relationships.ts` with `connections.ts` in the project structure. Update the description from `useRelationships` to `useConnections`.

- [ ] **Step 3: Commit**

```bash
git add backend/CLAUDE.md frontend/CLAUDE.md
git commit -m "docs: update CLAUDE.md for Connection model"
```

---

## Self-Review Checklist

1. **Spec coverage:**
   - Connection model with ConnectionType enum ✅ (Task 1)
   - CRUD with is_connected_as ✅ (Task 2)
   - Pydantic schemas ✅ (Task 3)
   - API routes ✅ (Task 4)
   - Sessions/metrics auth updated ✅ (Task 5)
   - Alembic migration ✅ (Task 6)
   - Old files deleted ✅ (Task 7)
   - Frontend types ✅ (Task 8)
   - Frontend API client ✅ (Task 9)
   - Frontend pages/components ✅ (Task 10)
   - Documentation ✅ (Task 11)

2. **Placeholder scan:**
   - No TBD, TODO, "implement later" found
   - All code steps have complete code blocks
   - All test code is provided

3. **Type consistency:**
   - `ConnectionType` used consistently: model enum, CRUD param, schema field, route handler
   - `from_user_id`/`to_user_id` used consistently across all layers
   - `is_connected_as` signature matches across CRUD definition and route usage
   - `ConnectionResponse` schema matches ORM model attributes
   - Frontend `Connection` type matches backend `ConnectionResponse`
