# Phase 1: Auth + PostgreSQL + User Profiles

**Date:** 2026-04-11
**Status:** Draft
**Scope:** Email/password auth, PostgreSQL persistence, user profiles, settings page

---

## Overview

Add authentication and user management to the existing FastAPI + Next.js platform. This is the foundation for all future SaaS features (coach-student linking, progress tracking, social features).

**What this phase includes:**
- PostgreSQL database with async SQLAlchemy 2.0
- Email/password registration and login
- JWT auth with refresh token rotation
- User profiles (name, avatar, bio, body params, preferences)
- Settings page on frontend

**What this phase excludes (future):**
- Coach/student roles and linking
- Progress tracking and video history
- OAuth providers (Telegram, Yandex, etc.)
- MFA / 2FA
- Billing and subscriptions
- MinIO/self-hosted storage migration

---

## Architecture

```
┌──────────────────────────────────────┐
│  Next.js Frontend (src/frontend/)    │
│  /login  /register  /profile/settings│
└──────────────┬───────────────────────┘
               │ HTTP (JWT Bearer)
┌──────────────▼───────────────────────┐
│  FastAPI Backend (src/backend/)      │
│  /api/auth/*   /api/users/me        │
├──────────┬──────────┬───────────────┤
│PostgreSQL│  Valkey  │  R2/S3        │
│ (users,  │ (cache,  │ (avatars,     │
│  tokens) │  sessions)│  uploads)    │
└──────────┴──────────┴───────────────┘
```

### Component boundaries

| Component | Responsibility | Depends on |
|-----------|---------------|------------|
| `src/backend/database.py` | Engine, session factory, get_db dependency | PostgreSQL via asyncpg |
| `src/backend/models/` | SQLAlchemy ORM models (User, RefreshToken) | database.py |
| `src/backend/auth/` | JWT creation/verification, password hashing, deps | models/, PyJWT, bcrypt |
| `src/backend/routes/auth.py` | /register, /login, /refresh, /logout | auth/, models/ |
| `src/backend/routes/users.py` | /me, /me/settings | auth/, models/ |
| `src/frontend/src/app/(auth)/` | Login, register pages | API client |
| `src/frontend/src/app/(dashboard)/profile/` | Profile, settings pages | API client |

---

## Database

### Driver: asyncpg

- 25-35% faster than psycopg3 for direct connections
- Full SQLAlchemy 2.0 async support
- No PgBouncer planned (single Podman instance) — no prepared statement issues

### Connection string

```
postgresql+asyncpg://user:pass@localhost:5432/skating_ml
```

### Session management

```python
# src/backend/database.py
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

engine = create_async_engine(settings.database_url, pool_size=10, max_overflow=20)
async_session = async_sessionmaker(engine, expire_on_commit=False)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session
```

Strict rules:
- Always use `async with` — never store sessions in request state
- `expire_on_commit=False` to allow accessing attributes after commit
- Pool size: 10 (adjust based on load testing)

### Migrations: Alembic

- Standard Alembic with autogenerate
- `alembic check` in CI (lefthook pre-commit)
- Always review with `alembic upgrade --sql` before applying
- Always write `downgrade()` immediately
- Short-lived branches (<3 days) to reduce merge conflicts

---

## ORM Models

### Design: SQLAlchemy 2.0 + separate Pydantic schemas

SQLAlchemy models for DB operations. Pydantic models for API request/response serialization. No SQLModel — async story is still maturing, and we need full control over complex queries.

### User model

```python
class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    hashed_password: Mapped[str]
    display_name: Mapped[str | None] = mapped_column(String(100))
    avatar_url: Mapped[str | None] = mapped_column(String(500))  # R2/S3 URL
    bio: Mapped[str | None] = mapped_column(Text)

    # Body params (for PhysicsEngine: CoM, Dempster tables)
    height_cm: Mapped[int | None] = mapped_column(SmallInt)  # user input
    weight_kg: Mapped[float | None] = mapped_column(Float)

    # Preferences
    language: Mapped[str] = mapped_column(String(10), default="ru")
    timezone: Mapped[str] = mapped_column(String(50), default="Europe/Moscow")
    theme: Mapped[str] = mapped_column(String(10), default="system")  # light/dark/system

    # Metadata
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())
```

### RefreshToken model

```python
class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)  # SHA-256 hash
    family_id: Mapped[uuid.UUID] = mapped_column(index=True)  # for rotation detection
    is_revoked: Mapped[bool] = mapped_column(default=False)
    expires_at: Mapped[datetime]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
```

### Indexes

- `users.email` — unique, for login lookups
- `refresh_tokens.token_hash` — unique, for token validation
- `refresh_tokens.family_id` — for batch revocation (stolen token detection)

---

## Authentication

### Strategy: Hand-rolled JWT (not fastapi-users)

Reasons:
- Full control over user model and auth flows
- No opinionated abstractions that fight with future coach/student roles
- Fewer dependencies, easier to audit for security
- fastapi-users is good for rapid prototyping but risky for production SaaS with custom requirements

### Libraries

| Library | Purpose | Version |
|---------|---------|---------|
| `PyJWT` | JWT encode/decode | latest |
| `passlib[bcrypt]` | Password hashing | bcrypt rounds=12 |
| `itsdangerous` | Email verification tokens | latest |

### Token design

**Access token** (JWT):
- Lifetime: 15 minutes
- Payload: `{"sub": user_id, "exp": ..., "type": "access"}`
- Signed with HS256 (server secret)
- Stateless — no DB lookup on every request (only on `/refresh`)

**Refresh token** (opaque):
- Lifetime: 7 days
- Stored as SHA-256 hash in `refresh_tokens` table
- Single-use — invalidated immediately on refresh
- `family_id` groups token chains for theft detection

### Refresh token rotation

```
Login → access(15m) + refresh(7d)
  │
  ├── Access expired → POST /refresh {refresh_token}
  │   └── Validate hash → revoke old → issue new pair
  │
  ├── Refresh reused after rotation → DELETE all tokens in family
  │   └── Force re-login (token theft detected)
  │
  └── Refresh expired → POST /login (re-authenticate)
```

### Password security

- `passlib.CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)`
- Pin `bcrypt<4.1` to avoid 72-byte limit ValueError on some deployments
- No password complexity rules beyond minimum length (8 chars) — let users choose

### FastAPI dependencies

```python
# Type aliases for clean injection
DbDep = Annotated[AsyncSession, Depends(get_db)]
CurrentUser = Annotated[User, Depends(get_current_user)]

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: DbDep,
) -> User:
    payload = jwt.decode(token, settings.secret_key, algorithms=["HS256"])
    user = await user_crud.get_by_id(db, uuid.UUID(payload["sub"]))
    if not user or not user.is_active:
        raise HTTPException(401)
    return user
```

### API endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/auth/register` | No | Create account |
| POST | `/api/auth/login` | No | Get tokens |
| POST | `/api/auth/refresh` | No (refresh token) | Rotate tokens |
| POST | `/api/auth/logout` | Yes | Revoke refresh token |
| GET | `/api/users/me` | Yes | Current user profile |
| PATCH | `/api/users/me` | Yes | Update profile |
| PATCH | `/api/users/me/settings` | Yes | Update preferences |

### Rate limiting

- Auth endpoints: 5 requests/minute per email (prevent brute force)
- Analysis endpoints: TBD (Phase 2+)
- Implementation: `slowapi` with Valkey backend
- Separate tiers for auth vs API vs static content

---

## Valkey Usage

**No ORM for Valkey.** Raw `redis-py` / `valkey-py` client only.

### Current uses (unchanged)
- arq job queue (task state, cancellation)

### New uses
- Rate limiting (`slowapi` stores counters in Valkey)
- Optional: session cache for blacklisted JWTs (if instant revocation needed)

### Key patterns
- Task state: `SETEX task:{id} 86400 {...}` (24h TTL, already implemented)
- Rate limit: `INCR ratelimit:{email}:{minute}` with 60s EXPIRE

---

## S3/R2 Storage

**No ORM.** Raw `boto3` client (already in `storage.py`).

### New use: Avatar uploads
- Key pattern: `avatars/{user_id}/{filename}`
- Max size: 2MB
- Accepted: JPEG, PNG, WebP
- Processing: resize to 256x256 before upload
- URL stored in `users.avatar_url`

---

## Infrastructure

### compose.yaml additions

```yaml
services:
  postgres:
    image: postgres:17-alpine
    environment:
      POSTGRES_DB: skating_ml
      POSTGRES_USER: skating
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-skating_dev}
    ports:
      - "127.0.0.1:5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U skating"]
      interval: 5s
      timeout: 3s
      retries: 5

  valkey:  # already exists

volumes:
  postgres_data:
```

### New Python dependencies

```
sqlalchemy[asyncio]>=2.0
asyncpg>=0.30
alembic>=1.14
PyJWT>=2.9
passlib[bcrypt]>=1.7
itsdangerous>=2.2
slowapi>=0.1
```

### Containerfile additions

- PostgreSQL client tools (for Alembic migrations in CI)
- No changes to runtime (DB is external service)

---

## Security considerations

1. **JWT secret**: Loaded from env var `JWT_SECRET_KEY`. Minimum 32 bytes. Generate with `python -c "import secrets; print(secrets.token_urlsafe(48))"`
2. **CORS**: Explicit origin list, never `allow_origins=["*"]` with `allow_credentials=True`
3. **Password hashing**: bcrypt with 12 rounds. Pin `bcrypt<4.1`.
4. **Refresh token storage**: SHA-256 hash only. Never store plaintext tokens in DB.
5. **SQL injection**: SQLAlchemy parameterized queries by default. No raw f-strings in SQL.
6. **Dependency auditing**: `pip-audit --strict` in CI.
7. **Email verification**: Not in Phase 1 scope. Add in Phase 2 if needed.

---

## Testing strategy

- Unit tests for auth logic (password hashing, JWT creation, token rotation)
- Integration tests for API endpoints (register, login, refresh, logout)
- Use pytest-asyncio for async test functions
- Test DB: separate PostgreSQL database or SQLite in-memory for unit tests
- Test factories: create test users via fixtures, not real registration

---

## What changes in existing code

### No changes to ML pipeline
The analysis pipeline (`src/pipeline.py`, `src/worker.py`, `src/web_helpers.py`) remains unchanged. Auth is a separate layer on top.

### Changes to FastAPI app
- Add auth middleware (optional for now — `Depends(get_current_user)` is sufficient)
- Add new route modules under `src/backend/routes/`
- Add database dependency injection

### Changes to compose.yaml
- Add PostgreSQL service

### Changes to config.py
- Add `database_url`, `jwt_secret_key` settings

### Changes to frontend
- New pages: `/login`, `/register`, `/profile`, `/profile/settings`
- API client: add auth headers (Bearer token)
- Token storage: httpOnly cookie or localStorage (decision needed at implementation)
- Protected routes: redirect to `/login` if no valid token

---

## Future considerations (not in scope)

- **MinIO**: Self-hosted S3-compatible storage to replace Cloudflare R2. Add when R2 costs become significant.
- **Coach/student roles**: Phase 2. Add `role` column to `users` table, role-based access control.
- **Email verification**: Phase 2. Add `is_verified` column, send verification email on registration.
- **MFA/2FA**: Far future. TOTP-based.
- **OAuth providers**: When political situation stabilizes or self-hosted IdP (Keycloak) is set up.
- **Billing**: Not planned yet. Usage-based pricing with Stripe or crypto.
