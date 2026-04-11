# Strava for Figure Skating — Coach Dashboard MVP

**Date:** 2026-04-11
**Status:** Draft
**Depends on:** [saas-auth-db-profiles-design.md](2026-04-11-saas-auth-db-profiles-design.md) (auth, Postgres, profiles)
**Scope:** Session history, progress tracking, coach-skater relationships, activity feed

---

## Overview

Transform the current one-shot analysis tool ("upload video → get results") into a Strava-like platform for figure skating. Every analysed video becomes a persistent "session" stored in the database. Users track progress over time, coaches manage students and diagnose problems automatically.

**What this phase includes:**
- Session persistence (save analysis results to Postgres)
- Personal records (PR) tracking
- Activity feed (skater view)
- Coach roster dashboard (coach view)
- Both roles fully responsive — coach may be on-ice with a phone
- Progress charts (metric trends over time)
- Automatic problem diagnostics
- Coach-skater relationships (flexible, no fixed roles)

**What this phase excludes:**
- Social features (feed of friends, kudos, comments)
- Assignments from coach to student
- Push notifications
- Reference comparison UI (DTW)
- Chat between coach and skater
- Billing / subscriptions
- Native mobile app (responsive web only, both roles mobile-friendly)

---

## Role System

### No fixed roles

Users have no `role` column. Role is defined by context of relationship. One person can be a coach to some users and a student to others simultaneously.

### Relationships

```
relationships — id, coach_id (FK users), skater_id (FK users),
                status (invited | active | ended),
                initiated_by (FK users),
                created_at, ended_at
```

**Lifecycle:**
1. Coach sends invite → `invited`
2. Skater accepts → `active`
3. Either party ends → `ended` (data preserved, access revoked)

**Constraints:**
- Partial unique index: `UNIQUE(coach_id, skater_id) WHERE status != 'ended'` — no duplicate active relationships (Alembic: use `postgresql_where` in `CreateIndex`)
- A user can appear as both `coach_id` and `skater_id` in different rows
- Re-inviting after `ended` creates a new row (old history preserved)

### Access control

| Action | Skater (own data) | Coach (student's data) | Stranger |
|--------|-------------------|----------------------|----------|
| View own sessions | Yes | — | No |
| View student sessions | — | Yes (active relationship) | No |
| Invite student | — | Yes | No |
| Accept/reject invite | Yes | — | — |
| End relationship | Yes | Yes | — |

---

## Data Model

### Sessions

```python
class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    element_type: Mapped[str] = mapped_column(String(50))  # waltz_jump, lutz, three_turn...
    video_url: Mapped[str | None]  # R2 URL to original video
    processed_video_url: Mapped[str | None]  # R2 URL to annotated video
    poses_url: Mapped[str | None]  # R2 URL to .npy
    csv_url: Mapped[str | None]  # R2 URL to biomechanics CSV
    status: Mapped[str] = mapped_column(String(20), default="uploading")
    # statuses: uploading | processing | done | failed
    error_message: Mapped[str | None]

    # Analysis results (denormalized for fast reads)
    phases: Mapped[dict | None]  # JSON: {takeoff: N, peak: N, landing: N, ...}
    recommendations: Mapped[list | None]  # JSON: ["text1", "text2", ...]
    overall_score: Mapped[float | None]

    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), index=True)
    processed_at: Mapped[datetime | None]
```

### Session Metrics

```python
class SessionMetric(Base):
    __tablename__ = "session_metrics"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    metric_name: Mapped[str] = mapped_column(String(100))  # airtime, max_height, landing_knee_stability...
    metric_value: Mapped[float]

    # PR tracking
    is_pr: Mapped[bool] = mapped_column(default=False)
    prev_best: Mapped[float | None]  # previous best value for this user+element+metric

    # Reference comparison
    reference_value: Mapped[float | None]  # from element_defs.py ideal ranges
    is_in_range: Mapped[bool | None]  # whether metric is within ideal range

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
```

**Unique constraint:** `UNIQUE(session_id, metric_name)`

### Metric Registry

Single source of truth for all metrics, shared between backend and frontend. Defined as a Python dict in `src/backend/metrics_registry.py` and exported as a static JSON endpoint `/api/v1/metrics/registry` for the frontend.

```python
METRIC_REGISTRY: dict[str, MetricDef] = {
    "airtime": MetricDef(
        name="airtime",
        label_ru="Время полёта",
        unit="s",
        format=".2f",
        direction="higher",  # higher is better
        element_types=["waltz_jump", "toe_loop", "flip", "salchow", "loop", "lutz", "axel"],
        ideal_range=(0.3, 0.7),
    ),
    "max_height": MetricDef(
        name="max_height",
        label_ru="Высота прыжка",
        unit="norm",  # normalized 0-1 (not meters — meters come from PhysicsEngine separately)
        format=".3f",
        direction="higher",
        element_types=["waltz_jump", "toe_loop", "flip", "salchow", "loop", "lutz", "axel"],
        ideal_range=(0.2, 0.5),
    ),
    "relative_jump_height": MetricDef(
        name="relative_jump_height",
        label_ru="Относительная высота",
        unit="ratio",
        format=".2f",
        direction="higher",
        element_types=["waltz_jump", "toe_loop", "flip", "salchow", "loop", "lutz", "axel"],
        ideal_range=(0.3, 1.5),
    ),
    "landing_knee_angle": MetricDef(
        name="landing_knee_angle",
        label_ru="Угол колена при приземлении",
        unit="deg",
        format=".0f",
        direction="higher",  # higher = more bent = better absorption
        element_types=["waltz_jump", "toe_loop", "flip", "salchow", "loop", "lutz", "axel"],
        ideal_range=(90, 130),
    ),
    "landing_knee_stability": MetricDef(
        name="landing_knee_stability",
        label_ru="Стабильность приземления",
        unit="score",
        format=".2f",
        direction="higher",
        element_types=["waltz_jump", "toe_loop", "flip", "salchow", "loop", "lutz", "axel"],
        ideal_range=(0.5, 1.0),
    ),
    "landing_trunk_recovery": MetricDef(
        name="landing_trunk_recovery",
        label_ru="Восстановление корпуса",
        unit="score",
        format=".2f",
        direction="higher",
        element_types=["waltz_jump", "toe_loop", "flip", "salchow", "loop", "lutz", "axel"],
        ideal_range=(0.5, 1.0),
    ),
    "arm_position_score": MetricDef(
        name="arm_position_score",
        label_ru="Контроль рук",
        unit="score",
        format=".2f",
        direction="higher",
        element_types=["waltz_jump", "toe_loop", "flip", "salchow", "loop", "lutz", "axel"],
        ideal_range=(0.6, 1.0),
    ),
    "rotation_speed": MetricDef(
        name="rotation_speed",
        label_ru="Скорость вращения",
        unit="deg/s",
        format=".0f",
        direction="higher",
        element_types=["waltz_jump", "toe_loop", "flip", "salchow", "loop", "lutz", "axel"],
        ideal_range=(300, 550),
    ),
    "knee_angle": MetricDef(
        name="knee_angle",
        label_ru="Угол колена",
        unit="deg",
        format=".0f",
        direction="higher",
        element_types=["three_turn"],
        ideal_range=(100, 140),
    ),
    "trunk_lean": MetricDef(
        name="trunk_lean",
        label_ru="Наклон корпуса",
        unit="deg",
        format=".1f",
        direction="higher",  # closer to 0 = better, but range is [-15, 20]
        element_types=["three_turn"],
        ideal_range=(-15, 20),
    ),
    "edge_change_smoothness": MetricDef(
        name="edge_change_smoothness",
        label_ru="Плавность смены ребра",
        unit="score",
        format=".2f",
        direction="higher",
        element_types=["three_turn"],
        ideal_range=(0.1, 0.5),
    ),
    "symmetry": MetricDef(
        name="symmetry",
        label_ru="Симметрия",
        unit="score",
        format=".2f",
        direction="higher",
        element_types=["waltz_jump", "toe_loop", "flip", "salchow", "loop", "lutz", "axel", "three_turn"],
        ideal_range=(0.6, 1.0),
    ),
}

@dataclass
class MetricDef:
    name: str
    label_ru: str
    unit: str          # "s", "deg", "score", "norm", "ratio", "deg/s"
    format: str        # Python format spec, e.g. ".2f"
    direction: str     # "higher" or "lower"
    element_types: list[str]
    ideal_range: tuple[float, float]
```

**Frontend consumption:** `GET /api/v1/metrics/registry` returns the full registry as JSON. Frontend uses it for:
- Metric dropdown labels (`label_ru`)
- Value formatting (`format`)
- PR direction logic
- Reference range display
- Filtering available metrics per element type

### Unit convention

- **Normalized metrics** (`unit="norm"`): 0-1 scale from `BiomechanicsAnalyzer`. Camera-dependent. Stored as-is.
- **Physical metrics** (`unit="s"`, `"deg"`, `"deg/s"`): Absolute values from pose geometry. Camera-independent for angles.
- **Score metrics** (`unit="score"`): 0-1 composite scores. Camera-independent.
- **PhysicsEngine metrics** (future): `jump_height_m` in meters from CoM parabolic fit. Not yet in `BiomechanicsAnalyzer` output — will be added when PhysicsEngine is wired into the session save flow.

### Overall score

`Session.overall_score` = fraction of metrics with `is_in_range=True`. Computed at session save time from the stored `session_metrics`. Range 0-1. Simple and interpretable: 1.0 = all metrics ideal, 0.5 = half need work.

### PR direction

Not all metrics are "higher is better." PR logic uses `direction` from `METRIC_REGISTRY`:
- `higher`: airtime, max_height, landing_knee_stability, etc. PR when `value > best`.
- `lower`: (none currently, reserved for future error-rate metrics). PR when `value < best`.

No separate direction config in `pr_tracker.py` — it reads from `METRIC_REGISTRY`.

### Indexes

- `sessions.user_id` — list user's sessions
- `sessions.created_at` — feed ordering
- `sessions.element_type` — filter by element
- `session_metrics.session_id` — join to session
- `session_metrics.metric_name` — filter by metric
- `session_metrics(session_id, metric_name)` — unique constraint (implicit index)
- Composite: `(user_id, element_type, created_at)` on sessions for trend queries

---

## Backend API

### Rewritten `/api/v1/` endpoints

Current ML endpoints (`/api/v1/detect`, `/api/v1/process/*`, `/api/v1/models`, `/api/v1/outputs/*`) are rewritten in-place. No separate `/api/v2/` — no production API to maintain backwards compatibility with. All endpoints require JWT auth (from auth spec).

#### Sessions

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/sessions` | Create session (upload initiated) |
| GET | `/api/v1/sessions` | List sessions (own, or student's if coach) |
| GET | `/api/v1/sessions/:id` | Get session detail with all metrics |
| PATCH | `/api/v1/sessions/:id` | Update session (element_type, notes) |
| DELETE | `/api/v1/sessions/:id` | Delete session (soft delete) |

**List sessions query params:**
- `user_id` — filter by user (coaches can pass student ID; skaters always see own)
- `element_type` — filter by element
- `limit`, `offset` — pagination
- `sort` — `created_at` (default) or `overall_score`

#### Metrics & Progress

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/metrics/trend` | Time series for a metric |
| GET | `/api/v1/metrics/prs` | List of personal records |
| GET | `/api/v1/metrics/diagnostics` | Automatic problem detection |
| GET | `/api/v1/metrics/registry` | Static metric definitions (label, unit, range, direction) |

**Trend query params:**
- `user_id` — whose metrics
- `element_type` — which element
- `metric_name` — which metric (e.g., `max_height`)
- `period` — `7d`, `30d`, `90d`, `all`

**Trend response:**
```json
{
  "metric_name": "max_height",
  "element_type": "lutz",
  "data_points": [
    {"date": "2026-04-01", "value": 0.38, "session_id": "...", "is_pr": false},
    {"date": "2026-04-03", "value": 0.42, "session_id": "...", "is_pr": true},
    {"date": "2026-04-05", "value": 0.41, "session_id": "...", "is_pr": false}
  ],
  "trend": "improving",  // improving | stable | declining
  "current_pr": 0.42,
  "reference_range": {"min": 0.35, "max": 0.55}
}
```

**Diagnostics response:**
```json
{
  "user_id": "...",
  "findings": [
    {
      "severity": "warning",
      "element": "lutz",
      "metric": "landing_knee_stability",
      "message": "Нестабильность приземления: ниже нормы в 4 из 5 последних сессий",
      "detail": "Среднее: 0.52 (норма: > 0.65), тренд: declining"
    },
    {
      "severity": "info",
      "element": "lutz",
      "metric": "max_height",
      "message": "Новый PR по высоте 3 дня назад!",
      "detail": "0.42м (предыдущий: 0.38м)"
    }
  ]
}
```

#### Relationships

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/relationships/invite` | Send invite (coach → skater) |
| POST | `/api/v1/relationships/:id/accept` | Accept invite |
| POST | `/api/v1/relationships/:id/end` | End relationship |
| GET | `/api/v1/relationships` | List relationships (as coach or skater) |
| GET | `/api/v1/relationships/pending` | Pending invites received |

---

## Pipeline Integration

### How sessions get created

The current ML pipeline (`process_video_pipeline` in `web_helpers.py`) runs async via arq worker. Extension point:

```
Current flow:
  Upload → R2 → arq task → ML pipeline → R2 output → done

New flow:
  Upload → R2 → arq task → ML pipeline → R2 output → Postgres session → done
                                                   └─ session_metrics
                                                   └─ PR check
                                                   └─ diagnostics update
```

After successful analysis:
1. Create `Session` row with `status=done`
2. Run `BiomechanicsAnalyzer` on the extracted poses (if not already run during processing)
3. Extract metrics from `MetricResult` list
4. For each metric:
   a. Query `session_metrics` for `user_id + element_type + metric_name` to find current best
   b. Check `METRIC_REGISTRY[metric_name].direction` — PR when value beats best in the correct direction
   c. If PR: set `is_pr=true`, store `prev_best=current_best`
   d. Compare against `ideal_range` from `METRIC_REGISTRY`, set `is_in_range`
5. Compute `overall_score` = count(`is_in_range=True`) / count(all metrics)
6. Store `phases` and `recommendations` as JSON on session

**Pipeline output extension:** `process_video_pipeline()` currently returns only file paths and stats. It must be extended to also return `list[MetricResult]`, `ElementPhase`, and `list[str]` (recommendations). This is done by calling `BiomechanicsAnalyzer.analyze()` and `Recommender.recommend()` at the end of the pipeline, before the Postgres save step.

### Body params in pipeline

User profile stores `height_cm` and `weight_kg` (from auth spec). These feed into `PhysicsEngine` for accurate CoM, moment of inertia, and parabolic jump height calculations.

**Flow:**
1. When creating a session (`POST /api/v1/sessions`), fetch user's body params from DB
2. Pass `height_cm`, `weight_kg` to `process_video_pipeline()` as optional params
3. Pipeline passes them to `PhysicsEngine(body_mass=weight_kg)` when computing physics metrics
4. Physics-derived metrics (e.g., `jump_height_m` from CoM trajectory) stored alongside biomechanical metrics

**Fallback:** If user hasn't set body params, use defaults (height=160cm, weight=60kg — typical figure skater). Display a subtle prompt in session detail: "Укажите рост и вес в профиле для более точных расчётов".

### Chunked upload

Phone videos are 100-500MB. Direct single-request upload fails on slow/unstable connections (ice rink WiFi).

**Approach:** S3 multipart upload via R2's S3-compatible API.

```
Client                          Server                    R2
  │                               │                        │
  ├─ POST /sessions (create) ────>│                        │
  ├─ InitiateMultipartUpload ────>├─ S3.create_multipart ─>│
  ├─ UploadPart (chunk 1) ──────>├─ S3.upload_part ───────>│
  ├─ UploadPart (chunk 2) ──────>├─ S3.upload_part ───────>│
  ├─ ...                         │                        │
  ├─ CompleteMultipartUpload ───>├─ S3.complete_multipart>│
  │                               │                        │
  ├─ POST /sessions/:id/process ─>├─ enqueue arq job       │
  │                               │                        │
```

**Implementation:**
- Backend: new endpoint `POST /api/v1/uploads/init` returns `upload_id` + pre-signed part URLs
- Client: splits file into 5MB chunks, uploads each to pre-signed URL in parallel (3 concurrent)
- Client: `POST /api/v1/uploads/:upload_id/complete` triggers pipeline
- Progress: each chunk upload updates local progress state (X/Y MB uploaded)
- Resume: if upload fails mid-way, client can retry individual chunks (S3 multipart supports this)

**Chunk size:** 5MB (balance between number of requests and retry granularity).

**Presigned URLs:** Server generates R2 pre-signed URLs for each part. No file data passes through the backend — direct client-to-R2 transfer.

---

## Frontend

### Tech stack (unchanged)

Next.js 16, App Router, TypeScript, Tailwind v4, shadcn/ui, next-intl, React Query, next-themes.

### Route structure

```
/                     → redirect by context:
                        logged out → /login
                        has active students → /dashboard
                        no students → /feed

# Skater routes
/feed                 → activity feed (session cards, newest first)
/upload               → simplified upload (pick video + element → submit)
/sessions/:id         → session detail (video, metrics, recommendations, PR delta)
/progress             → metric trend charts (element + metric selector)
/connections          → manage coach relationships

# Coach routes
/dashboard            → roster: list of students with mini-status cards
/students/:id         → student profile: progress charts + diagnostics
/students/:id/sessions/:id → student's session detail

# Shared
/profile              → own profile (settings, body params)
/settings             → preferences (language, theme, timezone)
/login                → auth
/register             → auth
```

### Responsive behavior

Both roles get the same responsive layout. No desktop-only or mobile-only views.

- Mobile (<768px): bottom tab bar. Tabs differ by context — skater sees Feed / Upload / Progress / Profile; coach sees Roster / Upload / Progress / Profile.
- Desktop (≥768px): sidebar navigation with all routes visible. Coach and skater routes coexist in the same nav — coach routes only shown when the user has active students.

### Activity Feed (`/feed`)

Vertical scroll of session cards. Each card shows:
- Element icon + name (e.g., "Лутц")
- Relative time ("2 часа назад")
- 1-2 key metrics with units ("Высота: 0.42м", "Полёт: 0.51с")
- PR badge if any metric is a personal record
- Color indicator: green if overall score in range, amber/red if problems detected
- Processing indicator (spinner) if `status != done`

Tap → `/sessions/:id`

### Session Detail (`/sessions/:id`)

Top section:
- Video player with annotated overlay (reuse current `VideoPlayer`)
- Element name + date

Metrics section:
- List of all metrics with values
- Color-coded: green (in range), amber (near boundary), red (out of range)
- PR delta shown next to value: "0.42м (+4см PR)"
- Reference range shown as subtle bar

Recommendations section:
- Rule-based recommendations from `recommender.py` (already in Russian)

### Progress (`/progress`)

Selectors at top:
- Element type dropdown (waltz_jump, lutz, flip...)
- Metric name dropdown (max_height, airtime, landing_knee_stability...)
- Period selector (7d, 30d, 90d, all)

Chart:
- Line chart: X = date, Y = metric value
- Data points with session links (tap point → session detail)
- PR points highlighted (gold marker)
- Reference range as shaded green band
- Trend line (simple linear regression)
- Trend label: "Улучшение" / "Стабильно" / "Ухудшение"

Chart library: Recharts (lightweight, React-native feel, already common in Next.js projects).

### Coach Dashboard (`/dashboard`)

Grid of student cards. Each card:
- Avatar + name
- Last session date ("3 дня назад")
- Session count this week
- Status dot: green (all metrics in range), amber (some warnings), red (problems detected)
- Quick action: tap → `/students/:id`

Sidebar link to manage invites (Connections).

### Student Profile (`/students/:id`)

Two tabs:

**Progress tab:**
- Same chart component as skater's `/progress`, but scoped to the student
- Default view: element with most sessions
- Quick switch between elements

**Diagnostics tab:**
- List of findings from `/api/v1/metrics/diagnostics`
- Grouped by severity (warning first, then info)
- Each finding: element icon, metric name, message, detail text
- Tap finding → filter progress chart to that element+metric

### Student Session Detail (`/students/:id/sessions/:id`)

Same as skater's session detail but read-only. No delete. Coach can see all metrics, recommendations, and PR history.

### Upload Flow (simplified)

Current flow has 5 steps. Simplify to 3:

1. **Pick video** — file picker or record in-browser
2. **Select element** — grid of element icons/tiles (waltz_jump, toe_loop, flip, salchow, loop, lutz, axel, three_turn). One tap.
3. **Submit** — show processing status, redirect to feed when done

#### In-browser recording

On the upload page, two entry points side by side: "Выбрать файл" and "Записать".

Recording uses native `MediaRecorder` API + `getUserMedia` — no libraries.

**Camera config:**
```typescript
{
  video: {
    facingMode: 'environment',  // rear camera
    width: { ideal: 1920 },
    fps: { ideal: 60 }          // needed for rotation analysis
  },
  audio: false                  // ice sound not needed for ML
}
```

**MIME type fallback chain:**
1. `video/webm; codecs=vp9` (Chrome, Firefox, Edge)
2. `video/mp4` (iOS Safari — webm not supported)
3. Check via `MediaRecorder.isTypeSupported()` before starting

**Recording UX:**
- Viewfinder showing live camera feed
- Red circle record button + elapsed timer
- Stop → preview with "Использовать" / "Перезаписать"
- Produced file feeds into the same upload pipeline as file picker

**Component:** `CameraRecorder` — self-contained, renders viewfinder + controls, emits `Blob` on stop.

Advanced options (gear icon → expand):
- Frame skip
- HUD layer
- Person selection (if multi-person video)

These default to sensible values and are hidden by default.

---

## Diagnostics Engine

### How automatic diagnostics work

Not ML — simple statistical aggregation on `session_metrics`.

**Rules (evaluated per user + element):**

1. **Consistently below range:**
   - Query last N sessions for `element_type + metric_name`
   - If >60% of values have `is_in_range=false` → warning
   - Message: "Метрика X ниже нормы в M из N последних сессий"

2. **Declining trend:**
   - Linear regression on last 5+ data points
   - If slope is negative and R² > 0.5 → warning
   - Message: "Метрика X ухудшается (тренд: declining)"

3. **Stagnation:**
   - If standard deviation < 5% of mean over 5+ sessions → info
   - Message: "Нет улучшений по метрике X за N сессий"

4. **New PR:**
   - Most recent session has `is_pr=true` → info
   - Message: "Новый PR по X!"

5. **High variability:**
   - If coefficient of variation > 20% over 5+ sessions → warning
   - Message: "Нестабильность: метрика X сильно колеблется"

**When diagnostics run:**
- On-demand: GET `/api/v1/metrics/diagnostics` computes for the requested user
- No pre-computation or caching in MVP (fast enough for <100 sessions per user)

---

## MVP Success Criteria

1. Skater uploads video, analysis runs, session is saved to Postgres
2. Session appears in activity feed with correct metrics
3. Progress chart shows trend for at least one metric
4. PR is detected and displayed correctly
5. Coach sees list of students with status indicators
6. Coach can view student's session history and progress charts
7. Diagnostics surface at least "consistently below range" and "new PR"
8. Coach-skater invite/accept/end flow works

---

## What changes in existing code

### Backend (minimal changes to ML pipeline)
- `web_helpers.py`: extend `process_video_pipeline()` to also return `list[MetricResult]`, `ElementPhase`, `list[str]`; add `save_session_to_db()` call after successful processing; accept `height_cm`, `weight_kg` params for PhysicsEngine
- `config.py`: add Postgres connection string to settings
- `compose.yaml`: add PostgreSQL service (already in auth spec)

### Backend (new modules)
- `src/backend/metrics_registry.py` — METRIC_REGISTRY (shared metric definitions)
- `src/backend/models/session.py` — Session, SessionMetric ORM models
- `src/backend/models/relationship.py` — Relationship ORM model
- `src/backend/routes/sessions.py` — session CRUD + PATCH endpoints
- `src/backend/routes/metrics.py` — trend, PR, diagnostics, registry endpoints
- `src/backend/routes/relationships.py` — invite/accept/end endpoints
- `src/backend/routes/uploads.py` — chunked upload (init, complete, presigned URLs)
- `src/backend/services/pr_tracker.py` — PR detection logic (reads direction from METRIC_REGISTRY)
- `src/backend/services/diagnostics.py` — diagnostic rules engine
- Alembic migration for new tables

### Frontend (significant redesign)
- New route structure (see above)
- New components: SessionCard, ActivityFeed, ProgressChart, RosterDashboard, DiagnosticsList, StudentCard, CameraRecorder, ChunkedUploader
- Modified upload flow (simplified + in-browser recording + chunked upload with progress)
- New API hooks (React Query)
- Metric formatting from registry (`label_ru`, `unit`, `format`)
- Responsive layout (bottom tabs mobile, sidebar desktop)
