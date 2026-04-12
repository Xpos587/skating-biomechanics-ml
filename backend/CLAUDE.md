# backend/CLAUDE.md — FastAPI Backend

## Project Structure

```
backend/
├── app/                              # Python package (backend.app.*)
│   ├── routes/                       # FastAPI routers
│   │   ├── auth.py                  # POST register/login/refresh
│   │   ├── users.py                 # GET/PATCH /users/me
│   │   ├── sessions.py              # CRUD /sessions
│   │   ├── metrics.py               # GET /metrics/trend, /prs, /diagnostics, /registry
│   │   ├── uploads.py               # POST init/chunk/complete
│   │   ├── detect.py                # POST/GET /detect (async queue)
│   │   ├── process.py               # POST /process (async queue)
│   │   ├── relationships.py         # GET/POST/PATCH /relationships
│   │   └── misc.py                  # Health check, etc.
│   ├── models/                       # SQLAlchemy ORM models
│   │   ├── base.py                  # Base, TimestampMixin
│   │   ├── user.py                  # User
│   │   ├── session.py               # Session, SessionMetric
│   │   └── relationship.py          # Coach-Skater relationship
│   ├── schemas.py                    # Pydantic request/response schemas (all in one file)
│   ├── crud/                         # Database CRUD operations
│   ├── services/                     # Business logic (diagnostics rules, etc.)
│   ├── config.py                     # Settings (Pydantic BaseSettings)
│   ├── storage.py                    # R2/S3 client
│   ├── task_manager.py               # Valkey task queue helpers
│   ├── database.py                   # SQLAlchemy async engine
│   ├── logging_config.py             # structlog configuration
│   ├── metrics_registry.py           # MetricDef definitions (12+ metrics, Russian labels, ideal ranges)
│   ├── auth/                         # JWT auth (deps.py — CurrentUser, DbDep)
│   └── main.py                       # FastAPI app factory
├── alembic/                          # Database migrations
├── tests/                            # Backend tests
└── pyproject.toml                    # Backend-only dependencies
```

## Architectural Constraint

**ZERO ML imports.** All ML runs in `ml/skating_ml/worker.py` (arq worker). Routes like `/detect` and `/process` enqueue jobs to Valkey; results are polled via status/result endpoints.

## API Routes

| Method | Path | Description |
|--------|------|-------------|
| POST | `/auth/register` | Create account |
| POST | `/auth/login` | Get JWT tokens |
| POST | `/auth/refresh` | Refresh access token |
| GET | `/users/me` | Current user profile |
| PATCH | `/users/me` | Update profile (name, bio, height, weight) |
| PATCH | `/users/me/settings` | Update settings (theme, language, timezone) |
| POST | `/sessions` | Create analysis session |
| GET | `/sessions` | List sessions (filter by user_id, element_type) |
| GET | `/sessions/{id}` | Get session with metrics |
| PATCH | `/sessions/{id}` | Update session |
| DELETE | `/sessions/{id}` | Soft delete session |
| GET | `/metrics/registry` | All metric definitions (static) |
| GET | `/metrics/trend` | Trend data points + linear regression |
| GET | `/metrics/prs` | Current personal records |
| GET | `/metrics/diagnostics` | Run diagnostic rules (stagnation, declining, etc.) |
| POST | `/uploads/init` | Start chunked upload (returns presigned URL) |
| POST | `/uploads/chunk` | Upload chunk to R2 |
| POST | `/uploads/complete` | Finalize upload, trigger processing |
| POST | `/detect` | Enqueue person detection job (async) |
| GET | `/detect/{task_id}/status` | Poll detection job status |
| GET | `/detect/{task_id}/result` | Get detection result |
| POST | `/process` | Start ML pipeline processing (async) |
| GET | `/relationships` | List relationships |
| POST | `/relationships/invite` | Invite skater |
| PATCH | `/relationships/{id}` | Accept/reject invitation |

## Auth Architecture

- **JWT**: access token (15min) + refresh token (7d), stored in localStorage
- **Cookie sync**: `sb_auth=1` cookie set by frontend for server-side gating
- **CurrentUser**: dependency injection via `backend.app.auth.deps` (reads JWT from Authorization header)
- **Coach access**: coaches can view students' sessions/metrics via `is_coach_for_student()` check

## Metrics System

`backend.app.metrics_registry` defines 12+ metrics per element:

| Metric | Unit | Direction | Elements |
|--------|------|-----------|----------|
| `airtime` | s | higher | jumps |
| `max_height` | norm | higher | jumps |
| `relative_jump_height` | ratio | higher | jumps |
| `landing_knee_angle` | deg | lower | jumps |
| `landing_knee_stability` | score | higher | jumps |
| `landing_trunk_recovery` | score | higher | jumps |
| `arm_position_score` | score | higher | jumps |
| `rotation_speed` | deg/s | higher | jumps |
| `knee_angle` | deg | lower | three_turn |
| `trunk_lean` | deg | lower | three_turn |
| `edge_change_smoothness` | score | higher | three_turn |
| `symmetry` | score | higher | all |

Each `MetricDef` has `label_ru`, `unit`, `format`, `direction`, `element_types`, `ideal_range`.

## Diagnostic Rules

`backend.app.services.diagnostics` implements 5 rules:
1. `check_consistently_below_range` — metric below ideal range in majority of sessions
2. `check_declining_trend` — linear regression shows decline (slope < 0, r² > 0.3)
3. `check_stagnation` — values flat with low variance
4. `check_new_pr` — latest session set a new personal record
5. `check_high_variability` — coefficient of variation too high

## Schemas

All schemas in `backend.app.schemas` (single file). Key types:
- `UserResponse`: id, email, display_name, avatar_url, bio, height_cm, weight_kg, language, timezone, theme
- `SessionResponse`: includes nested `SessionMetricResponse[]`
- `TrendResponse`: metric_name, data_points, trend (improving/stable/declining), current_pr, reference_range
- `DiagnosticsResponse`: user_id, findings[]
- `DetectQueueResponse`: task_id, video_key, status
- `DetectResultResponse`: persons, preview_image, video_key, auto_click, status
- `MLModelFlags`: depth, optical_flow, segment, foot_track, matting, inpainting

## Before Committing

1. **Tests**: `go-task test` or `uv run pytest backend/tests/`
2. **Type check**: `uv run basedpyright backend/app/`
3. **Lint**: `go-task lint` or `uv run ruff check backend/app/`
