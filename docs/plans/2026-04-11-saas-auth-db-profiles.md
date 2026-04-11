# SaaS Auth + DB + Profiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add email/password authentication, PostgreSQL persistence, and user profiles to the existing FastAPI + Next.js platform.

**Architecture:** SQLAlchemy 2.0 async with asyncpg driver, hand-rolled JWT auth with refresh token rotation, Alembic migrations. Backend adds `database.py`, `models/`, `auth/`, and new route modules. Frontend adds login/register pages, profile/settings pages, and auth-aware API client.

**Tech Stack:** PostgreSQL 17 Alpine, asyncpg, SQLAlchemy 2.0, Alembic, PyJWT, passlib[bcrypt], Next.js 16 App Router, TanStack Query

**Design decisions:**
- Token storage: localStorage (simpler, sufficient for now; httpOnly cookie is a future enhancement)
- CORS origins: loaded from `settings.cors_origins` (env var), not hardcoded
- Auth schemas in `src/backend/schemas_auth.py` (NOT `src/backend/schemas/` — conflicts with existing `schemas.py`)
- Frontend auth schemas in `src/frontend/src/lib/auth-schemas.ts` (NOT `src/frontend/src/lib/schemas/` — conflicts with existing `schemas.ts`)
- No OAuth2PasswordBearer `tokenUrl` (our login uses JSON body, not form-encoded)
- Avatar upload: deferred to Phase 2 (needs image processing + S3 integration)
- Rate limiting (slowapi): deferred to Phase 2 (needs Valkey integration testing)
- itsdangerous: not needed until email verification (Phase 2)

**Spec:** `data/specs/2026-04-11-saas-auth-db-profiles-design.md`

---

## File Structure

### New backend files

```
src/backend/
├── database.py              # Engine, session factory, get_db dependency
├── models/
│   ├── __init__.py          # Re-export Base, User, RefreshToken
│   ├── base.py              # DeclarativeBase with common mixins
│   ├── user.py              # User ORM model
│   └── refresh_token.py     # RefreshToken ORM model
├── auth/
│   ├── __init__.py          # Re-export deps, security
│   ├── security.py          # JWT encode/decode, password hashing, token creation
│   └── deps.py              # get_current_user, CurrentUser, DbDep type aliases
├── crud/
│   ├── __init__.py
│   ├── user.py              # get_by_id, get_by_email, create, update
│   └── refresh_token.py     # create, get_by_hash, revoke, revoke_family
├── schemas_auth.py          # Auth Pydantic schemas (separate file to avoid schemas/ conflict)
└── routes/
    ├── auth.py              # /api/auth/register, /login, /refresh, /logout
    └── users.py             # /api/users/me, /api/users/me/settings
```

### New frontend files

```
src/frontend/src/
├── lib/
│   ├── auth.ts              # Auth API client (login, register, refresh, logout)
│   └── auth-schemas.ts      # Auth Zod schemas (separate file to avoid schemas/ conflict)
├── app/
│   ├── (auth)/
│   │   ├── login/page.tsx   # Login page
│   │   └── register/page.tsx # Registration page
│   └── profile/
│       ├── page.tsx         # Profile view/edit
│       └── settings/page.tsx # Preferences (lang, tz, theme)
├── components/
│   └── auth-provider.tsx    # AuthProvider context + useAuth hook
```

### New infrastructure files

```
alembic.ini                  # Alembic config
alembic/
├── env.py                   # Async migration env
├── script.py.mako           # Migration template
└── versions/                # Migration scripts
```

### Modified files

```
pyproject.toml               # Add deps: sqlalchemy, asyncpg, alembic, PyJWT, passlib, email-validator
src/config.py                # Add database_url, jwt_secret_key, cors_origins, jwt settings
compose.yaml                 # Add postgres service
.env.example                 # Add DATABASE_URL, JWT_SECRET_KEY, CORS_ORIGINS
src/backend/main.py          # Register auth + users routers, lifespan with DB engine
src/frontend/src/lib/api.ts  # Add auth headers to all requests
src/frontend/src/components/app-nav.tsx  # Add Profile link, show user menu
src/frontend/src/app/providers.tsx  # Add AuthProvider
Taskfile.yml                 # Add db tasks (migrate, rollback, reset)
```

---

### Task 1: Add Python Dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add dependencies**

Add these to the `dependencies` list in `pyproject.toml`:

```toml
    "sqlalchemy[asyncio]>=2.0",
    "asyncpg>=0.30",
    "alembic>=1.14",
    "PyJWT>=2.9",
    "passlib[bcrypt]>=1.7",
    "bcrypt>=4.0,<4.1",
    "email-validator>=2.5",
```

- [ ] **Step 2: Install dependencies**

Run: `uv sync`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat(backend): add SQLAlchemy, asyncpg, auth dependencies"
```

---

### Task 2: Add PostgreSQL to Infrastructure

**Files:**
- Modify: `compose.yaml`
- Modify: `.env.example`

- [ ] **Step 1: Add postgres service to compose.yaml**

Replace the entire `compose.yaml` with:

```yaml
services:
  valkey:
    image: docker.io/valkey/valkey:alpine
    restart: unless-stopped
    ports:
      - "127.0.0.1:${VALKEY_HOST_PORT:-6379}:6379"
    healthcheck:
      test: ["CMD", "valkey-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5
      start_period: 5s
    volumes:
      - valkey-data:/data

  postgres:
    image: docker.io/postgres:17-alpine
    restart: unless-stopped
    ports:
      - "127.0.0.1:5432:5432"
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-skating_ml}
      POSTGRES_USER: ${POSTGRES_USER:-skating}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-skating_dev}
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U skating -d skating_ml"]
      interval: 5s
      timeout: 3s
      retries: 5
      start_period: 10s
    volumes:
      - postgres-data:/var/lib/postgresql/data

volumes:
  valkey-data:
  postgres-data:
```

- [ ] **Step 2: Update .env.example**

Add after the existing VALKEY section:

```env
# PostgreSQL
DATABASE_URL=postgresql+asyncpg://skating:skating_dev@localhost:5432/skating_ml

# JWT Authentication
JWT_SECRET_KEY=change-me-to-a-random-secret
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=15
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7
```

- [ ] **Step 3: Commit**

```bash
git add compose.yaml .env.example
git commit -m "feat(infra): add PostgreSQL 17 service to compose"
```

---

### Task 3: Add Database Settings to Config

**Files:**
- Modify: `src/config.py`

- [ ] **Step 1: Add database and JWT settings**

Add these fields to the `Settings` class in `src/config.py`, after `task_ttl_seconds`:

```python
    # PostgreSQL
    database_url: str = "postgresql+asyncpg://skating:skating_dev@localhost:5432/skating_ml"

    # JWT Authentication
    jwt_secret_key: str = "change-me-to-a-random-secret"
    jwt_access_token_expire_minutes: int = 15
    jwt_refresh_token_expire_days: int = 7

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
```

- [ ] **Step 2: Verify import**

The existing imports in `config.py` are `from pydantic_settings import BaseSettings` — this is sufficient. No new imports needed.

- [ ] **Step 3: Run typecheck**

Run: `uv run basedpyright --level error src/config.py`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add src/config.py
git commit -m "feat(config): add database_url and JWT settings"
```

---

### Task 4: Database Engine and Session

**Files:**
- Create: `src/backend/database.py`
- Create: `tests/backend/conftest.py`

- [ ] **Step 1: Write the test for database module**

Create `tests/backend/__init__.py` (empty) and `tests/backend/conftest.py`:

```python
"""Shared test fixtures for backend tests."""

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.backend.models import Base


@pytest_asyncio.fixture
async def db_engine():
    """Create a test database engine (SQLite in-memory for unit tests)."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    session_factory = async_sessionmaker(db_engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session
        await session.rollback()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/backend/conftest.py -v --no-header`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.backend.models'`

- [ ] **Step 3: Create models package**

Create `src/backend/models/__init__.py`:

```python
"""SQLAlchemy ORM models."""

from src.backend.models.base import Base
from src.backend.models.refresh_token import RefreshToken
from src.backend.models.user import User

__all__ = ["Base", "RefreshToken", "User"]
```

- [ ] **Step 4: Create base module**

Create `src/backend/models/base.py`:

```python
"""SQLAlchemy declarative base with common mixins."""

from datetime import datetime

from sqlalchemy import DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class TimestampMixin:
    """Mixin that adds created_at and updated_at columns."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(),
    )
```

- [ ] **Step 5: Create User model**

Create `src/backend/models/user.py`:

```python
"""User ORM model."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Float, SmallInt, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.backend.models.base import Base, TimestampMixin


class User(TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    display_name: Mapped[str | None] = mapped_column(String(100))
    avatar_url: Mapped[str | None] = mapped_column(String(500))
    bio: Mapped[str | None] = mapped_column(Text)

    # Body params (for PhysicsEngine: CoM, Dempster tables)
    height_cm: Mapped[int | None] = mapped_column(SmallInt)
    weight_kg: Mapped[float | None] = mapped_column(Float)

    # Preferences
    language: Mapped[str] = mapped_column(String(10), default="ru")
    timezone: Mapped[str] = mapped_column(String(50), default="Europe/Moscow")
    theme: Mapped[str] = mapped_column(String(10), default="system")

    # Metadata
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
```

Note: UUIDs stored as `String(36)` for asyncpg compatibility (native UUID type has edge cases across drivers).

- [ ] **Step 6: Create RefreshToken model**

Create `src/backend/models/refresh_token.py`:

```python
"""RefreshToken ORM model."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column

from src.backend.models.base import Base, TimestampMixin


class RefreshToken(TimestampMixin, Base):
    __tablename__ = "refresh_tokens"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), index=True,
    )
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    family_id: Mapped[str] = mapped_column(String(36), index=True)
    is_revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
```

- [ ] **Step 7: Create database.py**

Create `src/backend/database.py`:

```python
"""Async database engine and session factory."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.config import get_settings

settings = get_settings()

engine = create_async_engine(
    settings.database_url,
    pool_size=10,
    max_overflow=20,
    echo=settings.log_level == "DEBUG",
)

async_session_factory = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async database session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
```

- [ ] **Step 8: Add aiosqlite test dependency**

Run: `uv add --dev aiosqlite`

This is needed for SQLite in-memory async tests (production uses asyncpg).

- [ ] **Step 9: Run test to verify models work**

Create `tests/backend/test_models.py`:

```python
"""Tests for ORM models."""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.backend.models import Base, RefreshToken, User


@pytest.mark.asyncio
async def test_create_user(db_session: AsyncSession):
    """Test creating a user and reading it back."""
    user = User(
        email="test@example.com",
        hashed_password="hashed",
        display_name="Test User",
    )
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)

    result = await db_session.execute(select(User).where(User.email == "test@example.com"))
    fetched = result.scalar_one()

    assert fetched.email == "test@example.com"
    assert fetched.display_name == "Test User"
    assert fetched.is_active is True
    assert fetched.language == "ru"
    assert fetched.theme == "system"
    assert fetched.id is not None


@pytest.mark.asyncio
async def test_create_refresh_token(db_session: AsyncSession):
    """Test creating a refresh token linked to a user."""
    user = User(email="test@example.com", hashed_password="hashed")
    db_session.add(user)
    await db_session.flush()

    token = RefreshToken(
        user_id=user.id,
        token_hash="a" * 64,
        family_id="b" * 36,
        expires_at="2099-01-01",
    )
    db_session.add(token)
    await db_session.flush()
    await db_session.refresh(token)

    assert token.user_id == user.id
    assert token.is_revoked is False


@pytest.mark.asyncio
async def test_user_timestamps(db_session: AsyncSession):
    """Test that created_at is set automatically."""
    user = User(email="test@example.com", hashed_password="hashed")
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)

    assert user.created_at is not None
    assert user.updated_at is not None
```

- [ ] **Step 10: Run tests**

Run: `uv run pytest tests/backend/test_models.py -v`
Expected: All 3 tests PASS

- [ ] **Step 11: Commit**

```bash
git add src/backend/database.py src/backend/models/ tests/backend/ pyproject.toml uv.lock
git commit -m "feat(backend): add SQLAlchemy models, async engine, and session factory"
```

---

### Task 5: Alembic Setup

**Files:**
- Create: `alembic.ini`
- Create: `alembic/env.py`
- Create: `alembic/script.py.mako`
- Create: `alembic/versions/.gitkeep`
- Modify: `.gitignore`

- [ ] **Step 1: Initialize Alembic**

Run: `uv run alembic init alembic`

- [ ] **Step 2: Configure alembic.ini**

Replace `sqlalchemy.url` line in `alembic.ini`:

```ini
sqlalchemy.url = postgresql+asyncpg://skating:skating_dev@localhost:5432/skating_ml
```

Set `file_template` to include timestamps:

```ini
file_template = %%(year)d_%%(month).2d_%%(day).2d_%%(hour).2d%%(minute).2d-%%(rev)s_%%(slug)s
```

- [ ] **Step 3: Write async env.py**

Replace `alembic/env.py` with:

```python
"""Alembic async migration environment."""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from src.backend.models import Base
from src.config import get_settings

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Override sqlalchemy.url from app settings
config.set_main_option("sqlalchemy.url", get_settings().database_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in 'online' mode with async engine."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    import asyncio

    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

- [ ] **Step 4: Create versions directory**

Run: `mkdir -p alembic/versions && touch alembic/versions/.gitkeep`

- [ ] **Step 5: Add alembic to .gitignore**

Add this line to `.gitignore` (do NOT remove existing lines):

```
# Alembic (keep versions/, ignore __pycache__)
alembic/__pycache__/
```

- [ ] **Step 6: Generate initial migration**

Start postgres first: `podman compose up -d postgres`
Wait for healthy: `podman compose exec postgres pg_isready -U skating`

Run: `uv run alembic revision --autogenerate -m "initial users and refresh_tokens"`

Review the generated migration file in `alembic/versions/`. Verify it contains:
- `CREATE TABLE users` with all columns
- `CREATE TABLE refresh_tokens` with all columns and FK
- `CREATE INDEX` on `users.email`, `refresh_tokens.token_hash`, `refresh_tokens.family_id`
- `downgrade()` that drops both tables

- [ ] **Step 7: Apply migration**

Run: `uv run alembic upgrade head`
Expected: `INFO [alembic.runtime.migration] Running upgrade -> xxx, initial users and refresh_tokens`

- [ ] **Step 8: Add db tasks to Taskfile.yml**

Add these tasks to `Taskfile.yml` under the `# Python tasks` section:

```yaml
  db-migrate:
    desc: Run database migrations (Alembic upgrade)
    cmd: "{{.UV_RUN}} alembic upgrade head"

  db-rollback:
    desc: Rollback last database migration
    cmd: "{{.UV_RUN}} alembic downgrade -1"

  db-reset:
    desc: Reset database (drop all, recreate, migrate)
    cmds:
      - "{{.UV_RUN}} alembic downgrade base"
      - "{{.UV_RUN}} alembic upgrade head"

  db-migration:
    desc: Create a new autogenerate migration
    cmd: "{{.UV_RUN}} alembic revision --autogenerate -m '{{.migration_name}}'"
```

- [ ] **Step 9: Commit**

```bash
git add alembic.ini alembic/ Taskfile.yml .gitignore
git commit -m "feat(backend): setup Alembic async migrations with initial schema"
```

---

### Task 6: Auth Security Module

**Files:**
- Create: `src/backend/auth/__init__.py`
- Create: `src/backend/auth/security.py`
- Create: `tests/backend/test_security.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/backend/test_security.py`:

```python
"""Tests for auth security utilities."""

import time

import pytest

from src.backend.auth.security import (
    create_access_token,
    create_refresh_token,
    hash_password,
    verify_password,
)


def test_hash_and_verify_password():
    """Test password hashing and verification."""
    hashed = hash_password("secret123")
    assert hashed != "secret123"
    assert len(hashed) > 20
    assert verify_password("secret123", hashed) is True
    assert verify_password("wrong", hashed) is False


def test_hash_different_for_same_password():
    """Test that bcrypt produces different hashes for the same password (salt)."""
    h1 = hash_password("same")
    h2 = hash_password("same")
    assert h1 != h2


def test_verify_password_invalid_hash():
    """Test verification with garbage hash returns False."""
    assert verify_password("test", "not_a_real_hash") is False


def test_create_access_token():
    """Test JWT access token creation and decoding."""
    token = create_access_token(user_id="user-123")
    assert isinstance(token, str)
    assert len(token) > 20

    # Verify it's a valid JWT by checking structure (header.payload.signature)
    parts = token.split(".")
    assert len(parts) == 3


def test_create_access_token_custom_expiry():
    """Test access token with custom expiry."""
    token = create_access_token(user_id="user-123", expires_delta_seconds=1)
    # Token should still be valid immediately
    parts = token.split(".")
    assert len(parts) == 3

    # After 2 seconds it should be expired
    time.sleep(2)
    import jwt as pyjwt
    from src.config import get_settings

    settings = get_settings()
    with pytest.raises(pyjwt.ExpiredSignatureError):
        pyjwt.decode(token, settings.jwt_secret_key, algorithms=["HS256"])


def test_create_refresh_token():
    """Test refresh token is a random opaque string."""
    token = create_refresh_token()
    assert isinstance(token, str)
    assert len(token) == 64  # hex-encoded 32 bytes


def test_create_refresh_token_unique():
    """Test that two refresh tokens are unique."""
    t1 = create_refresh_token()
    t2 = create_refresh_token()
    assert t1 != t2


def test_hash_token():
    """Test token hashing for DB storage."""
    from src.backend.auth.security import hash_token

    token = create_refresh_token()
    hashed = hash_token(token)
    assert isinstance(hashed, str)
    assert len(hashed) == 64
    # Same token produces same hash
    assert hash_token(token) == hashed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/backend/test_security.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.backend.auth.security'`

- [ ] **Step 3: Create auth package**

Create `src/backend/auth/__init__.py`:

```python
"""Authentication utilities."""

from src.backend.auth.deps import CurrentUser, DbDep, get_current_user
from src.backend.auth.security import (
    create_access_token,
    create_refresh_token,
    hash_password,
    hash_token,
    verify_password,
)

__all__ = [
    "CurrentUser",
    "DbDep",
    "create_access_token",
    "create_refresh_token",
    "get_current_user",
    "hash_password",
    "hash_token",
    "verify_password",
]
```

Note: `deps.py` doesn't exist yet — we'll create it in Task 7. The `__init__.py` will error until then, which is fine.

- [ ] **Step 4: Implement security.py**

Create `src/backend/auth/security.py`:

```python
"""JWT creation, password hashing, and token utilities."""

from __future__ import annotations

import hashlib
import secrets

import jwt as pyjwt
from passlib.context import CryptContext

from src.config import get_settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a bcrypt hash."""
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception:
        return False


def create_access_token(user_id: str, expires_delta_seconds: int | None = None) -> str:
    """Create a short-lived JWT access token."""
    settings = get_settings()
    if expires_delta_seconds is None:
        expires_delta_seconds = settings.jwt_access_token_expire_minutes * 60

    payload = {
        "sub": user_id,
        "type": "access",
        "exp": int(__import__("time").time() + expires_delta_seconds),
    }
    return pyjwt.encode(payload, settings.jwt_secret_key, algorithm="HS256")


def create_refresh_token() -> str:
    """Create a random opaque refresh token (hex-encoded 32 bytes)."""
    return secrets.token_hex(32)


def hash_token(token: str) -> str:
    """Hash a refresh token for DB storage (SHA-256)."""
    return hashlib.sha256(token.encode()).hexdigest()
```

- [ ] **Step 5: Update auth __init__.py to not import deps yet**

Replace `src/backend/auth/__init__.py`:

```python
"""Authentication utilities."""

from src.backend.auth.security import (
    create_access_token,
    create_refresh_token,
    hash_password,
    hash_token,
    verify_password,
)

__all__ = [
    "create_access_token",
    "create_refresh_token",
    "hash_password",
    "hash_token",
    "verify_password",
]
```

We'll add the deps exports after Task 7.

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/backend/test_security.py -v`
Expected: All 8 tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/backend/auth/ tests/backend/test_security.py
git commit -m "feat(auth): add password hashing, JWT creation, and token utilities"
```

---

### Task 7: Auth Dependencies (get_current_user)

**Files:**
- Create: `src/backend/auth/deps.py`
- Create: `tests/backend/test_deps.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/backend/test_deps.py`:

```python
"""Tests for auth FastAPI dependencies."""

import pytest
from fastapi import HTTPException
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from src.backend.auth.deps import get_current_user
from src.backend.auth.security import create_access_token, hash_password
from src.backend.models import User


@pytest.mark.asyncio
async def test_get_current_user_valid(db_session: AsyncSession):
    """Test get_current_user returns user for valid token."""
    user = User(email="test@example.com", hashed_password=hash_password("pass"))
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)

    token = create_access_token(user_id=user.id)
    result = await get_current_user(token=token, db=db_session)

    assert result.id == user.id
    assert result.email == "test@example.com"


@pytest.mark.asyncio
async def test_get_current_user_invalid_token(db_session: AsyncSession):
    """Test get_current_user raises 401 for invalid token."""
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(token="invalid.jwt.token", db=db_session)
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_get_current_user_nonexistent_user(db_session: AsyncSession):
    """Test get_current_user raises 401 if user doesn't exist."""
    token = create_access_token(user_id="nonexistent-id")
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(token=token, db=db_session)
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_get_current_user_inactive_user(db_session: AsyncSession):
    """Test get_current_user raises 401 for inactive user."""
    user = User(email="test@example.com", hashed_password="hash", is_active=False)
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)

    token = create_access_token(user_id=user.id)
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(token=token, db=db_session)
    assert exc_info.value.status_code == 401
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/backend/test_deps.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.backend.auth.deps'`

- [ ] **Step 3: Create CRUD package**

Create `src/backend/crud/__init__.py`:

```python
"""CRUD operations for database models."""
```

Create `src/backend/crud/user.py`:

```python
"""User CRUD operations."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.backend.models.user import User


async def get_by_id(db: AsyncSession, user_id: str) -> User | None:
    """Get a user by ID."""
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


async def get_by_email(db: AsyncSession, email: str) -> User | None:
    """Get a user by email."""
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def create(db: AsyncSession, *, email: str, hashed_password: str, **kwargs) -> User:
    """Create a new user."""
    user = User(email=email, hashed_password=hashed_password, **kwargs)
    db.add(user)
    await db.flush()
    await db.refresh(user)
    return user


async def update(db: AsyncSession, user: User, **kwargs) -> User:
    """Update user fields."""
    for key, value in kwargs.items():
        if value is not None:
            setattr(user, key, value)
    db.add(user)
    await db.flush()
    await db.refresh(user)
    return user
```

Create `src/backend/crud/refresh_token.py`:

```python
"""RefreshToken CRUD operations."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.backend.models.refresh_token import RefreshToken


async def create(
    db: AsyncSession,
    *,
    user_id: str,
    token_hash: str,
    family_id: str,
    expires_at: datetime,
) -> RefreshToken:
    """Create a new refresh token."""
    token = RefreshToken(
        user_id=user_id,
        token_hash=token_hash,
        family_id=family_id,
        expires_at=expires_at,
    )
    db.add(token)
    await db.flush()
    await db.refresh(token)
    return token


async def get_by_hash(db: AsyncSession, token_hash: str) -> RefreshToken | None:
    """Get a refresh token by its hash."""
    result = await db.execute(select(RefreshToken).where(RefreshToken.token_hash == token_hash))
    return result.scalar_one_or_none()


async def revoke(db: AsyncSession, token: RefreshToken) -> None:
    """Revoke a single refresh token."""
    token.is_revoked = True
    db.add(token)
    await db.flush()


async def revoke_family(db: AsyncSession, family_id: str) -> int:
    """Revoke all tokens in a family (token theft detection)."""
    result = await db.execute(
        select(RefreshToken).where(
            RefreshToken.family_id == family_id,
            RefreshToken.is_revoked == False,  # noqa: E712
        )
    )
    tokens = result.scalars().all()
    count = 0
    for token in tokens:
        token.is_revoked = True
        db.add(token)
        count += 1
    await db.flush()
    return count


async def get_active_by_hash(
    db: AsyncSession, token_hash: str
) -> RefreshToken | None:
    """Get a non-revoked, non-expired refresh token by hash."""
    result = await db.execute(
        select(RefreshToken).where(
            RefreshToken.token_hash == token_hash,
            RefreshToken.is_revoked == False,  # noqa: E712
            RefreshToken.expires_at > datetime.now(timezone.utc),
        )
    )
    return result.scalar_one_or_none()
```

- [ ] **Step 4: Implement deps.py**

Create `src/backend/auth/deps.py`:

```python
"""FastAPI dependencies for authentication."""

from __future__ import annotations

from typing import Annotated

import jwt as pyjwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from src.backend.database import get_db
from src.backend.models.user import User
from src.config import get_settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

settings = get_settings()


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """Validate JWT and return the current user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = pyjwt.decode(token, settings.jwt_secret_key, algorithms=["HS256"])
        user_id: str | None = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except (pyjwt.InvalidTokenError, ValueError):
        raise credentials_exception

    from src.backend.crud.user import get_by_id

    user = await get_by_id(db, user_id)
    if user is None or not user.is_active:
        raise credentials_exception
    return user


# Type aliases for clean injection
DbDep = Annotated[AsyncSession, Depends(get_db)]
CurrentUser = Annotated[User, Depends(get_current_user)]
```

- [ ] **Step 5: Update auth __init__.py with deps exports**

Replace `src/backend/auth/__init__.py`:

```python
"""Authentication utilities."""

from src.backend.auth.deps import CurrentUser, DbDep, get_current_user
from src.backend.auth.security import (
    create_access_token,
    create_refresh_token,
    hash_password,
    hash_token,
    verify_password,
)

__all__ = [
    "CurrentUser",
    "DbDep",
    "create_access_token",
    "create_refresh_token",
    "get_current_user",
    "hash_password",
    "hash_token",
    "verify_password",
]
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/backend/test_deps.py -v`
Expected: All 4 tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/backend/auth/ src/backend/crud/ tests/backend/test_deps.py
git commit -m "feat(auth): add get_current_user dependency and CRUD operations"
```

---

### Task 8: Auth API Routes

**Files:**
- Create: `src/backend/routes/auth.py`
- Create: `src/backend/schemas_auth.py`
- Create: `tests/backend/test_auth_routes.py`

- [ ] **Step 1: Create auth Pydantic schemas**

Create `src/backend/schemas_auth.py` (separate file to avoid conflict with existing `schemas.py`):

```python
"""Auth-related Pydantic schemas."""

from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    display_name: str | None = Field(default=None, max_length=100)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    id: str
    email: str
    display_name: str | None
    avatar_url: str | None
    bio: str | None
    height_cm: int | None
    weight_kg: float | None
    language: str
    timezone: str
    theme: str
    is_active: bool
    created_at: str

    model_config = {"from_attributes": True}


class UpdateProfileRequest(BaseModel):
    display_name: str | None = Field(default=None, max_length=100)
    bio: str | None = None
    height_cm: int | None = Field(default=None, ge=50, le=250)
    weight_kg: float | None = Field(default=None, ge=20, le=300)


class UpdateSettingsRequest(BaseModel):
    language: str | None = Field(default=None, max_length=10)
    timezone: str | None = Field(default=None, max_length=50)
    theme: str | None = Field(default=None, pattern=r"^(light|dark|system)$")
```

Note: `EmailStr` requires `email-validator` package, already added in Task 1.

- [ ] **Step 2: Write the failing tests**

Create `tests/backend/test_auth_routes.py`:

```python
"""Tests for auth API routes."""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.backend.auth.security import create_access_token, hash_password
from src.backend.models import User


@pytest.fixture
def app():
    """Create test FastAPI app with auth routes."""
    from fastapi import FastAPI

    from src.backend.routes.auth import router

    app = FastAPI()
    app.include_router(router, prefix="/api/auth")
    return app


@pytest.fixture
async def client(app, db_session: AsyncSession):
    """Create test HTTP client with DB override."""
    from src.backend.database import get_db

    app.dependency_overrides[get_db] = lambda: db_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_register(client: AsyncClient, db_session: AsyncSession):
    """Test successful registration."""
    response = await client.post(
        "/api/auth/register",
        json={"email": "new@example.com", "password": "securepass123"},
    )
    assert response.status_code == 201
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"

    # Verify user was created
    from src.backend.crud.user import get_by_email
    from sqlalchemy import select

    result = await db_session.execute(select(User).where(User.email == "new@example.com"))
    user = result.scalar_one_or_none()
    assert user is not None
    assert user.hashed_password != "securepass123"


@pytest.mark.asyncio
async def test_register_duplicate_email(client: AsyncClient, db_session: AsyncSession):
    """Test registration with duplicate email returns 409."""
    # Create existing user
    user = User(email="exists@example.com", hashed_password="hash")
    db_session.add(user)
    await db_session.flush()

    response = await client.post(
        "/api/auth/register",
        json={"email": "exists@example.com", "password": "securepass123"},
    )
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_register_short_password(client: AsyncClient):
    """Test registration with short password returns 422."""
    response = await client.post(
        "/api/auth/register",
        json={"email": "new@example.com", "password": "short"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_login(client: AsyncClient, db_session: AsyncSession):
    """Test successful login."""
    user = User(email="login@example.com", hashed_password=hash_password("pass123"))
    db_session.add(user)
    await db_session.flush()

    response = await client.post(
        "/api/auth/login",
        json={"email": "login@example.com", "password": "pass123"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data


@pytest.mark.asyncio
async def test_login_wrong_password(client: AsyncClient, db_session: AsyncSession):
    """Test login with wrong password returns 401."""
    user = User(email="login@example.com", hashed_password=hash_password("correct"))
    db_session.add(user)
    await db_session.flush()

    response = await client.post(
        "/api/auth/login",
        json={"email": "login@example.com", "password": "wrong"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_login_nonexistent_email(client: AsyncClient):
    """Test login with nonexistent email returns 401."""
    response = await client.post(
        "/api/auth/login",
        json={"email": "nobody@example.com", "password": "pass123"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_refresh_tokens(client: AsyncClient, db_session: AsyncSession):
    """Test refresh token rotation."""
    user = User(email="refresh@example.com", hashed_password=hash_password("pass"))
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)

    # Login to get tokens
    login_resp = await client.post(
        "/api/auth/login",
        json={"email": "refresh@example.com", "password": "pass"},
    )
    tokens = login_resp.json()
    old_refresh = tokens["refresh_token"]

    # Use refresh token
    refresh_resp = await client.post(
        "/api/auth/refresh",
        json={"refresh_token": old_refresh},
    )
    assert refresh_resp.status_code == 200
    new_tokens = refresh_resp.json()
    assert new_tokens["refresh_token"] != old_refresh
    assert "access_token" in new_tokens

    # Old refresh token should be revoked
    second_refresh = await client.post(
        "/api/auth/refresh",
        json={"refresh_token": old_refresh},
    )
    assert second_refresh.status_code == 401
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/backend/test_auth_routes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.backend.routes.auth'`

- [ ] **Step 5: Implement auth routes**

Create `src/backend/routes/auth.py`:

```python
"""Auth API routes: register, login, refresh, logout."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.backend.auth.deps import DbDep
from src.backend.auth.security import (
    create_access_token,
    create_refresh_token,
    hash_password,
    hash_token,
    verify_password,
)
from src.backend.crud.refresh_token import create as create_refresh_token_crud
from src.backend.crud.refresh_token import get_active_by_hash, revoke
from src.backend.crud.user import create as create_user
from src.backend.crud.user import get_by_email
from src.backend.models.refresh_token import RefreshToken
from src.backend.schemas_auth import (
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
)
from src.config import get_settings

router = APIRouter(tags=["auth"])
settings = get_settings()


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest, db: DbDep):
    """Register a new user."""
    existing = await get_by_email(db, body.email)
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    user = await create_user(db, email=body.email, hashed_password=hash_password(body.password), display_name=body.display_name)

    access_token = create_access_token(user_id=user.id)
    refresh_token_str = create_refresh_token()
    family_id = str(uuid.uuid4())

    await create_refresh_token_crud(
        db,
        user_id=user.id,
        token_hash=hash_token(refresh_token_str),
        family_id=family_id,
        expires_at=datetime.now(timezone.utc) + timedelta(days=settings.jwt_refresh_token_expire_days),
    )

    return TokenResponse(access_token=access_token, refresh_token=refresh_token_str)


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: DbDep):
    """Authenticate and return tokens."""
    user = await get_by_email(db, body.email)
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    access_token = create_access_token(user_id=user.id)
    refresh_token_str = create_refresh_token()
    family_id = str(uuid.uuid4())

    await create_refresh_token_crud(
        db,
        user_id=user.id,
        token_hash=hash_token(refresh_token_str),
        family_id=family_id,
        expires_at=datetime.now(timezone.utc) + timedelta(days=settings.jwt_refresh_token_expire_days),
    )

    return TokenResponse(access_token=access_token, refresh_token=refresh_token_str)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(body: RefreshRequest, db: DbDep):
    """Rotate refresh token and issue new token pair."""
    token_hash = hash_token(body.refresh_token)
    existing = await get_active_by_hash(db, token_hash)

    if not existing:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired refresh token")

    # Revoke the old token (single-use)
    await revoke(db, existing)

    # Issue new pair in the same family
    access_token = create_access_token(user_id=existing.user_id)
    new_refresh_str = create_refresh_token()

    await create_refresh_token_crud(
        db,
        user_id=existing.user_id,
        token_hash=hash_token(new_refresh_str),
        family_id=existing.family_id,
        expires_at=datetime.now(timezone.utc) + timedelta(days=settings.jwt_refresh_token_expire_days),
    )

    return TokenResponse(access_token=access_token, refresh_token=new_refresh_str)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(body: RefreshRequest, db: DbDep):
    """Revoke a refresh token (client should also discard access token)."""
    token_hash = hash_token(body.refresh_token)
    existing = await get_active_by_hash(db, token_hash)
    if existing:
        await revoke(db, existing)
```

- [ ] **Step 6: Install httpx for test client**

Run: `uv add --dev httpx`

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/backend/test_auth_routes.py -v`
Expected: All 7 tests PASS

- [ ] **Step 8: Commit**

```bash
git add src/backend/routes/auth.py src/backend/schemas_auth.py tests/backend/test_auth_routes.py
git commit -m "feat(auth): add register, login, refresh, logout API routes"
```

---

### Task 9: User API Routes

**Files:**
- Create: `src/backend/routes/users.py`
- Create: `tests/backend/test_user_routes.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/backend/test_user_routes.py`:

```python
"""Tests for user API routes."""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.backend.auth.security import create_access_token, hash_password
from src.backend.models import User


@pytest.fixture
def app():
    from fastapi import FastAPI
    from src.backend.routes.users import router

    app = FastAPI()
    app.include_router(router, prefix="/api/users")
    return app


@pytest.fixture
async def client(app, db_session: AsyncSession):
    from src.backend.database import get_db

    app.dependency_overrides[get_db] = lambda: db_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.fixture
async def authed_user(db_session: AsyncSession) -> User:
    user = User(
        email="user@example.com",
        hashed_password=hash_password("pass"),
        display_name="Test User",
        bio="Skater",
        height_cm=175,
        weight_kg=70.0,
    )
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)
    return user


@pytest.fixture
def auth_headers(authed_user):
    token = create_access_token(user_id=authed_user.id)
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_get_me(client: AsyncClient, auth_headers):
    """Test GET /api/users/me returns current user."""
    response = await client.get("/api/users/me", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "user@example.com"
    assert data["display_name"] == "Test User"
    assert data["bio"] == "Skater"
    assert data["height_cm"] == 175
    assert data["weight_kg"] == 70.0


@pytest.mark.asyncio
async def test_get_me_unauthorized(client: AsyncClient):
    """Test GET /api/users/me without auth returns 401."""
    response = await client.get("/api/users/me")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_update_profile(client: AsyncClient, auth_headers):
    """Test PATCH /api/users/me updates profile fields."""
    response = await client.patch(
        "/api/users/me",
        json={"display_name": "New Name", "bio": "Updated bio", "height_cm": 180},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["display_name"] == "New Name"
    assert data["bio"] == "Updated bio"
    assert data["height_cm"] == 180


@pytest.mark.asyncio
async def test_update_settings(client: AsyncClient, auth_headers):
    """Test PATCH /api/users/me/settings updates preferences."""
    response = await client.patch(
        "/api/users/me/settings",
        json={"language": "en", "theme": "dark"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["language"] == "en"
    assert data["theme"] == "dark"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/backend/test_user_routes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.backend.routes.users'`

- [ ] **Step 3: Implement user routes**

Create `src/backend/routes/users.py`:

```python
"""User API routes: profile and settings."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from src.backend.auth.deps import CurrentUser, DbDep
from src.backend.crud.user import update
from src.backend.schemas_auth import (
    UpdateProfileRequest,
    UpdateSettingsRequest,
    UserResponse,
)

router = APIRouter(tags=["users"])


@router.get("/me", response_model=UserResponse)
async def get_me(user: CurrentUser):
    """Get current user profile."""
    return user


@router.patch("/me", response_model=UserResponse)
async def update_profile(body: UpdateProfileRequest, user: CurrentUser, db: DbDep):
    """Update current user profile."""
    updated = await update(
        db,
        user,
        display_name=body.display_name,
        bio=body.bio,
        height_cm=body.height_cm,
        weight_kg=body.weight_kg,
    )
    return updated


@router.patch("/me/settings", response_model=UserResponse)
async def update_settings(body: UpdateSettingsRequest, user: CurrentUser, db: DbDep):
    """Update current user preferences."""
    updated = await update(
        db,
        user,
        language=body.language,
        timezone=body.timezone,
        theme=body.theme,
    )
    return updated
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/backend/test_user_routes.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/backend/routes/users.py tests/backend/test_user_routes.py
git commit -m "feat(auth): add user profile and settings API routes"
```

---

### Task 10: Wire Auth into FastAPI App

**Files:**
- Modify: `src/backend/main.py`

- [ ] **Step 1: Register new routers in main.py**

Add imports and router registration to `src/backend/main.py`. The new imports go at the top with the existing ones:

```python
from src.backend.routes import auth, detect, models, process, users
```

Add these lines after the existing `app.include_router(process.router)`:

```python
app.include_router(auth.router)
app.include_router(users.router)
```

- [ ] **Step 2: Fix CORS (security requirement from spec)**

Replace the `allow_origins=["*"]` with config-driven list. First add import:

```python
from src.config import get_settings
```

Then replace the CORS middleware:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

- [ ] **Step 3: Run all backend tests**

Run: `uv run pytest tests/backend/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/backend/main.py
git commit -m "feat(backend): register auth and users routers, fix CORS origins"
```

---

### Task 11: Frontend Auth API Client

**Files:**
- Create: `src/frontend/src/lib/auth.ts`
- Create: `src/frontend/src/lib/auth-schemas.ts`

- [ ] **Step 1: Create auth Zod schemas**

Create `src/frontend/src/lib/auth-schemas.ts` (separate file to avoid conflict with existing `schemas.ts`):

```typescript
import { z } from "zod"

export const RegisterRequestSchema = z.object({
  email: z.string().email("Введите корректный email"),
  password: z.string().min(8, "Минимум 8 символов").max(128),
  display_name: z.string().max(100).optional(),
})

export const LoginRequestSchema = z.object({
  email: z.string().email(),
  password: z.string().min(1),
})

export const TokenResponseSchema = z.object({
  access_token: z.string(),
  refresh_token: z.string(),
  token_type: z.literal("bearer"),
})

export const UserResponseSchema = z.object({
  id: z.string(),
  email: z.string().email(),
  display_name: z.string().nullable(),
  avatar_url: z.string().nullable(),
  bio: z.string().nullable(),
  height_cm: z.number().int().nullable(),
  weight_kg: z.number().nullable(),
  language: z.string(),
  timezone: z.string(),
  theme: z.string(),
  is_active: z.boolean(),
  created_at: z.string(),
})

export const UpdateProfileRequestSchema = z.object({
  display_name: z.string().max(100).optional().nullable(),
  bio: z.string().optional().nullable(),
  height_cm: z.number().int().min(50).max(250).optional().nullable(),
  weight_kg: z.number().min(20).max(300).optional().nullable(),
})

export const UpdateSettingsRequestSchema = z.object({
  language: z.string().max(10).optional().nullable(),
  timezone: z.string().max(50).optional().nullable(),
  theme: z.enum(["light", "dark", "system"]).optional().nullable(),
})

export type RegisterRequest = z.infer<typeof RegisterRequestSchema>
export type LoginRequest = z.infer<typeof LoginRequestSchema>
export type TokenResponse = z.infer<typeof TokenResponseSchema>
export type UserResponse = z.infer<typeof UserResponseSchema>
export type UpdateProfileRequest = z.infer<typeof UpdateProfileRequestSchema>
export type UpdateSettingsRequest = z.infer<typeof UpdateSettingsRequestSchema>
```

- [ ] **Step 2: Create auth API client**

Create `src/frontend/src/lib/auth.ts`:

```typescript
import type { LoginRequest, RegisterRequest, TokenResponse, UserResponse } from "@/lib/auth-schemas"
import { TokenResponseSchema, UserResponseSchema } from "@/lib/auth-schemas"

const API_BASE = "/api"

const TOKEN_KEY = "access_token"
const REFRESH_KEY = "refresh_token"

export function getAccessToken(): string | null {
  if (typeof window === "undefined") return null
  return localStorage.getItem(TOKEN_KEY)
}

export function getRefreshToken(): string | null {
  if (typeof window === "undefined") return null
  return localStorage.getItem(REFRESH_KEY)
}

export function setTokens(access: string, refresh: string) {
  localStorage.setItem(TOKEN_KEY, access)
  localStorage.setItem(REFRESH_KEY, refresh)
}

export function clearTokens() {
  localStorage.removeItem(TOKEN_KEY)
  localStorage.removeItem(REFRESH_KEY)
}

export function authHeaders(): Record<string, string> {
  const token = getAccessToken()
  return token ? { Authorization: `Bearer ${token}` } : {}
}

export async function register(data: RegisterRequest): Promise<TokenResponse> {
  const res = await fetch(`${API_BASE}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Registration failed" }))
    throw new Error(err.detail)
  }
  const json = await res.json()
  return TokenResponseSchema.parse(json)
}

export async function login(data: LoginRequest): Promise<TokenResponse> {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  })
  if (!res.ok) {
    throw new Error("Неверный email или пароль")
  }
  const json = await res.json()
  return TokenResponseSchema.parse(json)
}

export async function refreshToken(refresh: string): Promise<TokenResponse> {
  const res = await fetch(`${API_BASE}/auth/refresh`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ refresh_token: refresh }),
  })
  if (!res.ok) {
    clearTokens()
    throw new Error("Сессия истекла")
  }
  const json = await res.json()
  return TokenResponseSchema.parse(json)
}

export async function logout(): Promise<void> {
  const refresh = getRefreshToken()
  if (refresh) {
    await fetch(`${API_BASE}/auth/logout`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: refresh }),
    }).catch(() => {})
  }
  clearTokens()
}

export async function fetchMe(): Promise<UserResponse> {
  const res = await fetch(`${API_BASE}/users/me`, {
    headers: { ...authHeaders() },
  })
  if (!res.ok) throw new Error("Unauthorized")
  const json = await res.json()
  return UserResponseSchema.parse(json)
}

export async function updateProfile(data: Partial<RegisterRequest> & { height_cm?: number; weight_kg?: number; bio?: string }): Promise<UserResponse> {
  const res = await fetch(`${API_BASE}/users/me`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(data),
  })
  if (!res.ok) throw new Error("Update failed")
  const json = await res.json()
  return UserResponseSchema.parse(json)
}

export async function updateSettings(data: { language?: string; timezone?: string; theme?: string }): Promise<UserResponse> {
  const res = await fetch(`${API_BASE}/users/me/settings`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(data),
  })
  if (!res.ok) throw new Error("Update failed")
  const json = await res.json()
  return UserResponseSchema.parse(json)
}
```

- [ ] **Step 3: Update api.ts to include auth headers**

Modify `src/frontend/src/lib/api.ts`. Update each fetch call to include auth headers. Add this import at the top:

```typescript
import { authHeaders } from "@/lib/auth"
```

Then update each `fetch()` call to merge auth headers. For example, `getModels`:

```typescript
export async function getModels(): Promise<ModelStatus[]> {
  const res = await fetch(`${API_BASE}/models`, { headers: { ...authHeaders() } })
  if (!res.ok) throw new Error("Failed to fetch model status")
  return res.json()
}
```

Apply the same pattern to all functions that need auth (process, detect, etc.). Public endpoints (health, models list) can remain without auth.

- [ ] **Step 4: Run frontend typecheck**

Run: `cd src/frontend && bun run typecheck`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
git add src/frontend/src/lib/auth.ts src/frontend/src/lib/auth-schemas.ts src/frontend/src/lib/api.ts
git commit -m "feat(frontend): add auth API client and Zod schemas"
```

---

### Task 12: Frontend Login and Register Pages

**Files:**
- Create: `src/frontend/src/app/(auth)/login/page.tsx`
- Create: `src/frontend/src/app/(auth)/register/page.tsx`
- Create: `src/frontend/src/app/(auth)/layout.tsx`

- [ ] **Step 1: Create auth layout**

Create `src/frontend/src/app/(auth)/layout.tsx`:

```tsx
export default function AuthLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-[calc(100vh-8rem)] items-center justify-center">
      <div className="w-full max-w-sm">{children}</div>
    </div>
  )
}
```

- [ ] **Step 2: Create login page**

Create `src/frontend/src/app/(auth)/login/page.tsx`:

```tsx
"use client"

import Link from "next/link"
import { useRouter } from "next/navigation"
import { type FormEvent, useState } from "react"
import { toast } from "sonner"
import { useAuth } from "@/components/auth-provider"

export default function LoginPage() {
  const router = useRouter()
  const { login } = useAuth()
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setLoading(true)
    try {
      await login(email, password)
      toast.success("Вход выполнен")
      router.push("/")
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Ошибка входа")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="space-y-2 text-center">
        <h1 className="text-2xl font-bold">Вход</h1>
        <p className="text-sm text-muted-foreground">Введите email и пароль</p>
      </div>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="space-y-2">
          <label htmlFor="email" className="text-sm font-medium">
            Email
          </label>
          <input
            id="email"
            type="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
            placeholder="you@example.com"
          />
        </div>
        <div className="space-y-2">
          <label htmlFor="password" className="text-sm font-medium">
            Пароль
          </label>
          <input
            id="password"
            type="password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
            placeholder="••••••••"
          />
        </div>
        <button
          type="submit"
          disabled={loading}
          className="w-full rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
        >
          {loading ? "Вход..." : "Войти"}
        </button>
      </form>
      <p className="text-center text-sm text-muted-foreground">
        Нет аккаунта?{" "}
        <Link href="/register" className="text-primary hover:underline">
          Зарегистрироваться
        </Link>
      </p>
    </div>
  )
}
```

- [ ] **Step 3: Create register page**

Create `src/frontend/src/app/(auth)/register/page.tsx`:

```tsx
"use client"

import Link from "next/link"
import { useRouter } from "next/navigation"
import { type FormEvent, useState } from "react"
import { toast } from "sonner"
import { useAuth } from "@/components/auth-provider"

export default function RegisterPage() {
  const router = useRouter()
  const { register } = useAuth()
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [displayName, setDisplayName] = useState("")
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setLoading(true)
    try {
      await register(email, password, displayName || undefined)
      toast.success("Аккаунт создан")
      router.push("/")
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Ошибка регистрации")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="space-y-2 text-center">
        <h1 className="text-2xl font-bold">Регистрация</h1>
        <p className="text-sm text-muted-foreground">Создайте аккаунт для начала работы</p>
      </div>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="space-y-2">
          <label htmlFor="email" className="text-sm font-medium">
            Email
          </label>
          <input
            id="email"
            type="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
            placeholder="you@example.com"
          />
        </div>
        <div className="space-y-2">
          <label htmlFor="name" className="text-sm font-medium">
            Имя (необязательно)
          </label>
          <input
            id="name"
            type="text"
            value={displayName}
            onChange={(e) => setDisplayName(e.target.value)}
            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
            placeholder="Ваше имя"
          />
        </div>
        <div className="space-y-2">
          <label htmlFor="password" className="text-sm font-medium">
            Пароль
          </label>
          <input
            id="password"
            type="password"
            required
            minLength={8}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
            placeholder="Минимум 8 символов"
          />
        </div>
        <button
          type="submit"
          disabled={loading}
          className="w-full rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
        >
          {loading ? "Создание..." : "Создать аккаунт"}
        </button>
      </form>
      <p className="text-center text-sm text-muted-foreground">
        Уже есть аккаунт?{" "}
        <Link href="/login" className="text-primary hover:underline">
          Войти
        </Link>
      </p>
    </div>
  )
}
```

- [ ] **Step 4: Run frontend build**

Run: `cd src/frontend && bun run build`
Expected: Build succeeds

- [ ] **Step 5: Commit**

```bash
git add src/frontend/src/app/\(auth\)/
git commit -m "feat(frontend): add login and register pages"
```

---

### Task 13: Frontend Profile and Settings Pages

**Files:**
- Create: `src/frontend/src/app/profile/page.tsx`
- Create: `src/frontend/src/app/profile/settings/page.tsx`

- [ ] **Step 1: Create profile page**

Create `src/frontend/src/app/profile/page.tsx`:

```tsx
"use client"

import { useRouter } from "next/navigation"
import { type FormEvent, useState } from "react"
import { toast } from "sonner"
import { useAuth } from "@/components/auth-provider"

export default function ProfilePage() {
  const { user, isLoading, logout } = useAuth()
  const router = useRouter()

  const [displayName, setDisplayName] = useState(user?.display_name ?? "")
  const [bio, setBio] = useState(user?.bio ?? "")
  const [height, setHeight] = useState(user?.height_cm?.toString() ?? "")
  const [weight, setWeight] = useState(user?.weight_kg?.toString() ?? "")
  const [saving, setSaving] = useState(false)

  if (isLoading) return <div className="text-center text-muted-foreground">Загрузка...</div>
  if (!user) {
    router.push("/login")
    return null
  }

  async function handleSave(e: FormEvent) {
    e.preventDefault()
    setSaving(true)
    try {
      const { updateProfile } = await import("@/lib/auth")
      const updated = await updateProfile({
        display_name: displayName || undefined,
        bio: bio || undefined,
        height_cm: height ? Number.parseInt(height) : undefined,
        weight_kg: weight ? Number.parseFloat(weight) : undefined,
      })
      toast.success("Профиль обновлён")
    } catch {
      toast.error("Ошибка сохранения")
    } finally {
      setSaving(false)
    }
  }

  async function handleLogout() {
    await logout()
    router.push("/")
  }

  return (
    <div className="mx-auto max-w-lg space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Профиль</h1>
        <button
          type="button"
          onClick={handleLogout}
          className="text-sm text-muted-foreground hover:text-foreground"
        >
          Выйти
        </button>
      </div>

      <form onSubmit={handleSave} className="space-y-4">
        <div className="space-y-2">
          <label htmlFor="email" className="text-sm font-medium">
            Email
          </label>
          <input
            id="email"
            type="email"
            value={user.email}
            disabled
            className="w-full rounded-md border border-input bg-muted px-3 py-2 text-sm"
          />
        </div>

        <div className="space-y-2">
          <label htmlFor="name" className="text-sm font-medium">
            Имя
          </label>
          <input
            id="name"
            type="text"
            value={displayName}
            onChange={(e) => setDisplayName(e.target.value)}
            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
          />
        </div>

        <div className="space-y-2">
          <label htmlFor="bio" className="text-sm font-medium">
            О себе
          </label>
          <textarea
            id="bio"
            value={bio}
            onChange={(e) => setBio(e.target.value)}
            rows={3}
            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <label htmlFor="height" className="text-sm font-medium">
              Рост (см)
            </label>
            <input
              id="height"
              type="number"
              value={height}
              onChange={(e) => setHeight(e.target.value)}
              min={50}
              max={250}
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
            />
          </div>
          <div className="space-y-2">
            <label htmlFor="weight" className="text-sm font-medium">
              Вес (кг)
            </label>
            <input
              id="weight"
              type="number"
              value={weight}
              onChange={(e) => setWeight(e.target.value)}
              min={20}
              max={300}
              step={0.1}
              className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
            />
          </div>
        </div>

        <button
          type="submit"
          disabled={saving}
          className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
        >
          {saving ? "Сохранение..." : "Сохранить"}
        </button>
      </form>
    </div>
  )
}
```

- [ ] **Step 2: Create settings page**

Create `src/frontend/src/app/profile/settings/page.tsx`:

```tsx
"use client"

import { useRouter } from "next/navigation"
import { type FormEvent, useState } from "react"
import { toast } from "sonner"
import { useAuth } from "@/components/auth-provider"

const LANGUAGES = [
  { value: "ru", label: "Русский" },
  { value: "en", label: "English" },
]

const THEMES = [
  { value: "system", label: "Системная" },
  { value: "light", label: "Светлая" },
  { value: "dark", label: "Тёмная" },
]

export default function SettingsPage() {
  const { user, isLoading } = useAuth()
  const router = useRouter()

  const [language, setLanguage] = useState(user?.language ?? "ru")
  const [timezone, setTimezone] = useState(user?.timezone ?? "Europe/Moscow")
  const [theme, setTheme] = useState(user?.theme ?? "system")
  const [saving, setSaving] = useState(false)

  if (isLoading) return <div className="text-center text-muted-foreground">Загрузка...</div>
  if (!user) {
    router.push("/login")
    return null
  }

  async function handleSave(e: FormEvent) {
    e.preventDefault()
    setSaving(true)
    try {
      const { updateSettings } = await import("@/lib/auth")
      await updateSettings({ language, timezone, theme })
      toast.success("Настройки сохранены")
    } catch {
      toast.error("Ошибка сохранения")
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="mx-auto max-w-lg space-y-6">
      <h1 className="text-2xl font-bold">Настройки</h1>

      <form onSubmit={handleSave} className="space-y-4">
        <div className="space-y-2">
          <label htmlFor="language" className="text-sm font-medium">
            Язык
          </label>
          <select
            id="language"
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
          >
            {LANGUAGES.map((l) => (
              <option key={l.value} value={l.value}>
                {l.label}
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-2">
          <label htmlFor="timezone" className="text-sm font-medium">
            Часовой пояс
          </label>
          <input
            id="timezone"
            type="text"
            value={timezone}
            onChange={(e) => setTimezone(e.target.value)}
            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
            placeholder="Europe/Moscow"
          />
        </div>

        <div className="space-y-2">
          <label htmlFor="theme" className="text-sm font-medium">
            Тема
          </label>
          <div className="flex gap-2">
            {THEMES.map((t) => (
              <button
                key={t.value}
                type="button"
                onClick={() => setTheme(t.value)}
                className={`rounded-md border px-4 py-2 text-sm ${theme === t.value ? "border-primary bg-primary/10" : "border-input"}`}
              >
                {t.label}
              </button>
            ))}
          </div>
        </div>

        <button
          type="submit"
          disabled={saving}
          className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
        >
          {saving ? "Сохранение..." : "Сохранить"}
        </button>
      </form>
    </div>
  )
}
```

- [ ] **Step 3: Run frontend build**

Run: `cd src/frontend && bun run build`
Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add src/frontend/src/app/profile/
git commit -m "feat(frontend): add profile and settings pages"
```

---

### Task 14: Update Navigation with Auth State

**Files:**
- Modify: `src/frontend/src/components/app-nav.tsx`
- Modify: `src/frontend/src/app/providers.tsx`

- [ ] **Step 1: Create auth-aware nav**

Replace `src/frontend/src/components/app-nav.tsx`:

```tsx
"use client"

import { Activity, LogIn, LogOut, Settings, Trophy, User } from "lucide-react"
import Link from "next/link"
import { usePathname, useRouter } from "next/navigation"
import { ThemeToggle } from "@/components/theme-toggle"
import { useAuth } from "@/components/auth-provider"

const navItems = [
  { href: "/", label: "Анализ", icon: Activity },
  { href: "/training", label: "Тренировки", icon: Trophy },
] as const

export function AppNav() {
  const pathname = usePathname()
  const router = useRouter()
  const { isAuthenticated, user, logout } = useAuth()

  async function handleLogout() {
    await logout()
    router.push("/")
  }

  return (
    <nav className="flex items-center gap-1">
      {navItems.map((item) => {
        const Icon = item.icon
        const isActive = pathname === item.href
        return (
          <Link
            key={item.href}
            href={item.href}
            className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm transition-colors hover:bg-muted ${isActive ? "bg-muted font-medium" : "text-muted-foreground"}`}
          >
            <Icon className="h-4 w-4" />
            <span className="hidden md:inline">{item.label}</span>
          </Link>
        )
      })}

      {isAuthenticated ? (
        <>
          <Link
            href="/profile"
            className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm transition-colors hover:bg-muted ${pathname === "/profile" ? "bg-muted font-medium" : "text-muted-foreground"}`}
          >
            <User className="h-4 w-4" />
            <span className="hidden md:inline">{user?.display_name ?? "Профиль"}</span>
          </Link>
          <button
            type="button"
            onClick={handleLogout}
            className="flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:bg-muted"
          >
            <LogOut className="h-4 w-4" />
          </button>
        </>
      ) : (
        <Link
          href="/login"
          className="flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm transition-colors hover:bg-muted text-muted-foreground"
        >
          <LogIn className="h-4 w-4" />
          <span className="hidden md:inline">Войти</span>
        </Link>
      )}

      <ThemeToggle />
    </nav>
  )
}
```

- [ ] **Step 2: Update providers to include auth**

Replace `src/frontend/src/app/providers.tsx`:

```tsx
"use client"

import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { ThemeProvider } from "next-themes"
import { useState, type ReactNode } from "react"
import { AuthProvider } from "@/components/auth-provider"

export function Providers({ children }: { children: ReactNode }) {
  const [queryClient] = useState(() => new QueryClient())

  return (
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
      <QueryClientProvider client={queryClient}>
        <AuthProvider>{children}</AuthProvider>
      </QueryClientProvider>
    </ThemeProvider>
  )
}
```

- [ ] **Step 3: Create auth provider component**

Create `src/frontend/src/components/auth-provider.tsx`:

```tsx
"use client"

import { createContext, useContext, useEffect, useState, type ReactNode } from "react"
import type { UserResponse } from "@/lib/schemas/auth"
import * as auth from "@/lib/auth"

interface AuthContextValue {
  user: UserResponse | null
  isLoading: boolean
  isAuthenticated: boolean
  login: (email: string, password: string) => Promise<void>
  register: (email: string, password: string, displayName?: string) => Promise<void>
  logout: () => Promise<void>
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<UserResponse | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const token = auth.getAccessToken()
    if (!token) {
      setIsLoading(false)
      return
    }

    auth
      .fetchMe()
      .then(setUser)
      .catch(() => {
        auth.clearTokens()
        setUser(null)
      })
      .finally(() => setIsLoading(false))
  }, [])

  async function login(email: string, password: string) {
    const tokens = await auth.login({ email, password })
    auth.setTokens(tokens.access_token, tokens.refresh_token)
    const u = await auth.fetchMe()
    setUser(u)
  }

  async function register(email: string, password: string, displayName?: string) {
    const tokens = await auth.register({ email, password, display_name: displayName })
    auth.setTokens(tokens.access_token, tokens.refresh_token)
    const u = await auth.fetchMe()
    setUser(u)
  }

  async function logout() {
    await auth.logout()
    setUser(null)
  }

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        isAuthenticated: !!user,
        login,
        register,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error("useAuth must be used within AuthProvider")
  return ctx
}
```

Note: Login, register, profile, and settings pages already use `useAuth` from `@/components/auth-provider` (created in Tasks 12-13 above). No further changes needed to those pages.

- [ ] **Step 4: Run frontend build**

Run: `cd src/frontend && bun run build`
Expected: Build succeeds

- [ ] **Step 5: Commit**

```bash
git add src/frontend/src/components/auth-provider.tsx src/frontend/src/components/app-nav.tsx src/frontend/src/app/providers.tsx src/frontend/src/app/\(auth\)/ src/frontend/src/app/profile/
git commit -m "feat(frontend): add AuthProvider, update nav with login/logout, auth-aware pages"
```

---

### Task 15: Final Integration Test

**Files:**
- Modify: `tests/backend/test_auth_routes.py`

- [ ] **Step 1: Run all backend tests**

Run: `uv run pytest tests/backend/ -v`
Expected: All tests PASS

- [ ] **Step 2: Run all existing tests (no regressions)**

Run: `uv run pytest tests/ -v -m "not slow" --tb=short`
Expected: All tests PASS (existing ML tests should not be affected)

- [ ] **Step 3: Run frontend typecheck**

Run: `cd src/frontend && bun run typecheck`
Expected: No errors

- [ ] **Step 4: Run frontend build**

Run: `cd src/frontend && bun run build`
Expected: Build succeeds

- [ ] **Step 5: Start services and manual smoke test**

Run: `podman compose up -d`
Run: `uv run alembic upgrade head`
Run: `uv run uvicorn src.backend.main:app --reload --port 8000`
Run: `cd src/frontend && bun run dev`

Manual verification:
1. Open `http://localhost:3000`
2. Nav shows "Войти" link
3. Click "Войти" → login page
4. Click "Зарегистрироваться" → register page
5. Register with test email
6. Redirected to home, nav shows user name + "Профиль"
7. Go to `/profile` → edit name, bio, height, weight → save
8. Go to `/profile/settings` → change theme → save
9. Click logout → nav shows "Войти" again

- [ ] **Step 6: Commit any fixes**

If smoke test reveals issues, fix and commit:
```bash
git add -A
git commit -m "fix: address smoke test issues"
```
