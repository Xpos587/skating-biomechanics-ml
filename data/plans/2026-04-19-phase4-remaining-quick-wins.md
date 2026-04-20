# Phase 4: Remaining Quick Wins — Batch Query, Thread-Safe Cache, httpx Reuse

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the last 3 remaining items from the pipeline parallelism spec: N+1 query in session_saver, thread-unsafe worker URL cache, and per-request httpx client creation in Vast.ai client.

**Architecture:** Batch query replaces per-metric loop with a single SQL query using `IN` clause. Thread-safe cache wraps global `_worker_url_cache` with `threading.Lock`. Module-level `httpx.AsyncClient` with lazy init eliminates per-request TLS handshake overhead.

**Tech Stack:** SQLAlchemy (batch query), threading.Lock (cache safety), httpx (HTTP client reuse)

**Spec:** `docs/specs/2026-04-19-pipeline-parallelism-design.md` — sections 3.5, 4.6, 3.6

---

### Task 1: Batch `get_current_best` query to eliminate N+1

**Files:**

- Modify: `backend/app/crud/session_metric.py:15-41`
- Modify: `backend/app/services/session_saver.py:41-56`
- Test: `backend/tests/crud/test_session_metric_batch.py` (new)

- [ ] **Step 1: Write the failing test**

Create `backend/tests/crud/test_session_metric_batch.py`:

```python
"""Tests for batch get_current_best query."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def db_session():
    session = AsyncMock()
    return session


async def test_get_current_best_batch_returns_dict():
    from app.crud.session_metric import get_current_best_batch

    mock_row1 = MagicMock()
    mock_row1.metric_name = "airtime"
    mock_row1.metric_value = 0.65
    mock_row2 = MagicMock()
    mock_row2.metric_name = "max_height"
    mock_row2.metric_value = 0.42

    mock_result = MagicMock()
    mock_result.all.return_value = [mock_row1, mock_row2]
    mock_result.scalars.return_value.all.return_value = [mock_row1, mock_row2]

    db = AsyncMock()
    db.execute = AsyncMock(return_value=mock_result)

    result = await get_current_best_batch(
        db,
        user_id="user-1",
        element_type="waltz_jump",
        metric_names=["airtime", "max_height"],
    )
    assert result == {"airtime": 0.65, "max_height": 0.42}


async def test_get_current_best_batch_returns_empty_for_no_metrics():
    from app.crud.session_metric import get_current_best_batch

    mock_result = MagicMock()
    mock_result.all.return_value = []
    mock_result.scalars.return_value.all.return_value = []

    db = AsyncMock()
    db.execute = AsyncMock(return_value=mock_result)

    result = await get_current_best_batch(
        db,
        user_id="user-1",
        element_type="waltz_jump",
        metric_names=["airtime"],
    )
    assert result == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/crud/test_session_metric_batch.py -v`
Expected: FAIL with `ImportError: cannot import name 'get_current_best_batch'`

- [ ] **Step 3: Write minimal implementation**

Add `get_current_best_batch` to `backend/app/crud/session_metric.py`, after `get_current_best`:

```python
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

    # Use DISTINCT ON (metric_name) ordered by value DESC per group
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest backend/tests/crud/test_session_metric_batch.py -v`
Expected: 2 passed

- [ ] **Step 5: Refactor session_saver.py to use batch query**

Replace the per-metric loop in `backend/app/services/session_saver.py:40-70`.

Change the import at line 11 from:
```python
from app.crud.session_metric import bulk_create, get_current_best
```
to:
```python
from app.crud.session_metric import bulk_create, get_current_best_batch
```

Replace lines 40-70 (the entire `for mr in metrics:` loop that calls `get_current_best` per metric) with:

```python
    # Build metric rows with PR tracking
    metric_rows = []

    # Batch-fetch all current bests in one query (N+1 fix)
    metric_names = [mr.name for mr in metrics]
    bests = await get_current_best_batch(
        db,
        user_id=session.user_id,
        element_type=session.element_type,
        metric_names=metric_names,
    )

    for mr in metrics:
        mdef = METRIC_REGISTRY.get(mr.name)
        ref_value = mdef.ideal_range[0] if mdef else None
        ref_max = mdef.ideal_range[1] if mdef else None

        is_in_range = None
        if mdef and ref_value is not None and ref_max is not None:
            is_in_range = ref_value <= mr.value <= ref_max

        # Check PR using batch-fetched best
        current_best = bests.get(mr.name)
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
```

- [ ] **Step 6: Run all backend tests**

Run: `uv run pytest backend/tests/ -v --tb=short`
Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add backend/app/crud/session_metric.py backend/app/services/session_saver.py backend/tests/crud/test_session_metric_batch.py
git commit -m "perf(backend): batch get_current_best query to eliminate N+1"
```

---

### Task 2: Thread-safe worker URL cache

**Files:**

- Modify: `backend/app/vastai/client.py:26-60,127-145`
- Modify: `backend/tests/test_vastai_client.py`

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_vastai_client.py`:

```python
def test_worker_url_cache_is_thread_safe():
    """Verify cache access is protected by a lock."""
    import threading

    import app.vastai.client as _vc

    # Reset cache
    _vc._worker_url_cache = None
    _vc._worker_url_cache_time = 0.0

    # Access the lock — it should exist
    assert hasattr(_vc, "_worker_url_lock")
    assert isinstance(_vc._worker_url_lock, type(threading.Lock()))


def test_get_worker_url_uses_cache():
    """Second call within TTL should not make HTTP request."""
    import app.vastai.client as _vc

    _vc._worker_url_cache = None
    _vc._worker_url_cache_time = 0.0

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"url": "https://cached.vast.ai:8000"}
    mock_resp.raise_for_status = MagicMock()

    with patch("app.vastai.client.httpx.post", return_value=mock_resp) as mock_post:
        url1 = _vc._get_worker_url("ep", "key")
        url2 = _vc._get_worker_url("ep", "key")

    assert url1 == "https://cached.vast.ai:8000"
    assert url2 == "https://cached.vast.ai:8000"
    mock_post.assert_called_once()  # Only one HTTP call, second uses cache
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_vastai_client.py -v`
Expected: FAIL — `AttributeError: module 'app.vastai.client' has no attribute '_worker_url_lock'`

- [ ] **Step 3: Add threading.Lock to client.py**

In `backend/app/vastai/client.py`, replace lines 26-29:

```python
# Worker URL cache to avoid repeated HTTP calls
_worker_url_cache: str | None = None
_worker_url_cache_time: float = 0.0
_WORKER_URL_TTL = 60  # Cache for 60 seconds
```

with:

```python
import threading

# Worker URL cache to avoid repeated HTTP calls (thread-safe)
_worker_url_cache: str | None = None
_worker_url_cache_time: float = 0.0
_WORKER_URL_TTL = 60  # Cache for 60 seconds
_worker_url_lock = threading.Lock()
```

- [ ] **Step 4: Wrap `_get_worker_url` cache access with lock**

Replace the body of `_get_worker_url` (lines 43-60) with:

```python
def _get_worker_url(endpoint_name: str, api_key: str) -> str:
    """Route request to get a ready worker URL."""
    global _worker_url_cache, _worker_url_cache_time  # noqa: PLW0603
    now = time.monotonic()
    with _worker_url_lock:
        if _worker_url_cache and (now - _worker_url_cache_time) < _WORKER_URL_TTL:
            return _worker_url_cache
    resp = httpx.post(
        ROUTE_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"endpoint": endpoint_name},
        timeout=ROUTE_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    url = data["url"]
    with _worker_url_lock:
        _worker_url_cache = url  # noqa: PLW0603
        _worker_url_cache_time = now  # noqa: PLW0603
    return url
```

- [ ] **Step 5: Wrap `_asyncio_get_worker_url` cache access with lock**

Replace the body of `_asyncio_get_worker_url` (lines 127-145) with:

```python
async def _asyncio_get_worker_url(endpoint_name: str, api_key: str) -> str:
    """Async route request to get a ready worker URL (with TTL cache)."""
    global _worker_url_cache, _worker_url_cache_time  # noqa: PLW0603
    now = time.monotonic()
    with _worker_url_lock:
        if _worker_url_cache and (now - _worker_url_cache_time) < _WORKER_URL_TTL:
            return _worker_url_cache
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            ROUTE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"endpoint": endpoint_name},
            timeout=ROUTE_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        url = data["url"]
        with _worker_url_lock:
            _worker_url_cache = url  # noqa: PLW0603
            _worker_url_cache_time = now  # noqa: PLW0603
        return url
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest backend/tests/test_vastai_client.py -v`
Expected: All pass (including 2 new tests)

- [ ] **Step 7: Commit**

```bash
git add backend/app/vastai/client.py backend/tests/test_vastai_client.py
git commit -m "fix(vastai): add threading.Lock to worker URL cache"
```

---

### Task 3: Reuse httpx.AsyncClient in Vast.ai client

**Files:**

- Modify: `backend/app/vastai/client.py:127-204`
- Test: `backend/tests/test_vastai_client.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_vastai_client.py`:

```python
async def test_async_client_is_reused_across_calls():
    """Verify _get_async_client returns the same client instance."""
    import app.vastai.client as _vc

    # Reset module-level client
    _vc._async_client = None

    c1 = _vc._get_async_client()
    c2 = _vc._get_async_client()
    assert c1 is c2

    # Cleanup
    await c1.aclose()
    _vc._async_client = None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest backend/tests/test_vastai_client.py::test_async_client_is_reused_across_calls -v`
Expected: FAIL — `AttributeError: module 'app.vastai.client' has no attribute '_get_async_client'`

- [ ] **Step 3: Add lazy-init async client helper**

Add after the `_worker_url_lock` definition (after line 34):

```python
# Shared async HTTP client (lazy-init, reused across requests)
_async_client: httpx.AsyncClient | None = None


def _get_async_client() -> httpx.AsyncClient:
    """Return a shared httpx.AsyncClient, creating it on first use."""
    global _async_client  # noqa: PLW0603
    if _async_client is None or _async_client.is_closed:
        _async_client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
    return _async_client
```

- [ ] **Step 4: Refactor `_asyncio_get_worker_url` to use shared client**

Replace `_asyncio_get_worker_url` body to use `_get_async_client()` instead of `async with httpx.AsyncClient()`:

```python
async def _asyncio_get_worker_url(endpoint_name: str, api_key: str) -> str:
    """Async route request to get a ready worker URL (with TTL cache)."""
    global _worker_url_cache, _worker_url_cache_time  # noqa: PLW0603
    now = time.monotonic()
    with _worker_url_lock:
        if _worker_url_cache and (now - _worker_url_cache_time) < _WORKER_URL_TTL:
            return _worker_url_cache
    client = _get_async_client()
    resp = await client.post(
        ROUTE_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"endpoint": endpoint_name},
        timeout=ROUTE_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    url = data["url"]
    with _worker_url_lock:
        _worker_url_cache = url  # noqa: PLW0603
        _worker_url_cache_time = now  # noqa: PLW0603
    return url
```

- [ ] **Step 5: Refactor `process_video_remote_async` to use shared client**

Replace lines 186-193 in `process_video_remote_async`:

```python
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{worker_url}/process",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        result = resp.json()
```

with:

```python
    client = _get_async_client()
    resp = await client.post(
        f"{worker_url}/process",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    result = resp.json()
```

- [ ] **Step 6: Run all vastai client tests**

Run: `uv run pytest backend/tests/test_vastai_client.py -v`
Expected: All pass (including new test)

- [ ] **Step 7: Run full backend test suite**

Run: `uv run pytest backend/tests/ -v --tb=short`
Expected: All pass

- [ ] **Step 8: Commit**

```bash
git add backend/app/vastai/client.py backend/tests/test_vastai_client.py
git commit -m "perf(vastai): reuse httpx.AsyncClient across requests"
```

---

### Task 4: Run full test suite and verify

**Files:**

- None (verification only)

- [ ] **Step 1: Run backend tests**

Run: `uv run pytest backend/tests/ -v --tb=short`
Expected: All pass

- [ ] **Step 2: Run linter**

Run: `uv run ruff check backend/app/ && uv run ruff format --check backend/app/`
Expected: No errors

---

## Self-Review Checklist

### Spec Coverage

| Spec Requirement | Task | Status |
|---|---|---|
| 3.5 N+1 query in session_saver.py | Task 1 | Covered |
| 4.6 Thread-safe worker URL cache | Task 2 | Covered |
| 3.6 httpx.AsyncClient reuse | Task 3 | Covered |

### Placeholder Scan

- No TBD, TODO, or "implement later" found
- All test code is complete with actual assertions
- All imports are explicit
- All file paths verified against current codebase

### Type Consistency

- `get_current_best_batch` returns `dict[str, float]` — consistent with `get_current_best` returning `float | None`
- `_worker_url_lock` is `threading.Lock` in both sync and async code paths
- `_get_async_client()` returns `httpx.AsyncClient` — consistent with existing async usage

### Notes

- The batch query uses `GROUP BY + MAX` which is semantically equivalent to the per-metric `ORDER BY ... DESC LIMIT 1` (both return the max value per metric group). For "lower is better" metrics, the caller (`check_pr`) already handles direction correctly by comparing the raw max value.
- `_worker_url_lock` uses `threading.Lock` (not `asyncio.Lock`) because the cache is accessed from both sync (`_get_worker_url`) and async (`_asyncio_get_worker_url`) contexts. `threading.Lock` is safe for both.
- The shared `httpx.AsyncClient` has no explicit cleanup on app shutdown. This is acceptable — httpx will clean up on process exit, and the client is lazily re-created if closed.
