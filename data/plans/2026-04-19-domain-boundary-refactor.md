# Domain Boundary Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move orchestrator code (worker, vastai client) from `ml/` to `backend/` so `ml/` becomes a pure ML library with zero infrastructure imports.

**Architecture:** `ml/src/vastai/client.py` and `ml/src/worker.py` are NOT ML code — they are HTTP routing and job orchestration. They move to `backend/app/`. `ml/` drops the `skating-backend` dependency entirely. `backend/` adds `skating-ml` as a dependency (types + vastai dispatch only, no pipeline internals).

**Tech Stack:** Python, arq, httpx, pydantic-settings, pytest

---

### Task 1: Move vastai/client.py to backend

**Rationale:** `vastai/client.py` is a pure HTTP client that routes requests to Vast.ai GPU workers. It imports only `httpx` and `backend.app.config`. It has zero ML imports. It belongs in `backend/`.

**Files:**
- Move: `ml/src/vastai/client.py` → `backend/app/vastai/client.py`
- Create: `backend/app/vastai/__init__.py`
- Move: `ml/tests/test_vastai_client.py` → `backend/tests/test_vastai_client.py`
- Modify: `ml/src/worker.py:210` (import path)

- [ ] **Step 1: Create backend/app/vastai/__init__.py**

```python
```

- [ ] **Step 2: Move ml/src/vastai/client.py to backend/app/vastai/client.py**

Copy the entire file. Fix the import:

```python
# Old:
from backend.app.config import get_settings

# New:
from app.config import get_settings
```

No other changes — all other imports are stdlib (`logging`, `time`, `dataclasses`) or third-party (`httpx`).

- [ ] **Step 3: Delete ml/src/vastai/ directory**

```bash
rm -rf ml/src/vastai/
```

- [ ] **Step 4: Move ml/tests/test_vastai_client.py to backend/tests/test_vastai_client.py**

Copy the file. Fix all import paths:

```python
# Old:
from src.vastai.client import VastResult, _get_worker_url

# New:
from app.vastai.client import VastResult, _get_worker_url
```

Also fix the mock patches:

```python
# Old:
with patch("src.vastai.client.httpx.post", ...)
with patch("src.vastai.client.get_settings") as mock_settings:
import src.vastai.client as _vc
_vc._worker_url_cache = None

# New:
with patch("app.vastai.client.httpx.post", ...)
with patch("app.vastai.client.get_settings") as mock_settings:
import app.vastai.client as _vc
_vc._worker_url_cache = None
```

- [ ] **Step 5: Delete old test file**

```bash
rm ml/tests/test_vastai_client.py
```

- [ ] **Step 6: Update worker.py import (line 210)**

In `ml/src/worker.py`, change the lazy import inside `process_video_task`:

```python
# Old (line 210):
from src.vastai.client import process_video_remote_async

# New:
from backend.app.vastai.client import process_video_remote_async
```

Wait — worker.py is moving to backend/ in Task 2, so this import will become a relative import. Don't change it now; Task 2 will handle this.

- [ ] **Step 7: Run backend tests**

Run: `uv run pytest backend/tests/test_vastai_client.py -v`
Expected: 3 tests PASS

- [ ] **Step 8: Commit**

```bash
git add backend/app/vastai/ backend/tests/test_vastai_client.py
git rm ml/src/vastai/ ml/tests/test_vastai_client.py
git commit -m "refactor: move vastai client from ml/ to backend/"
```

---

### Task 2: Move worker.py to backend

**Rationale:** `worker.py` is an arq job orchestrator. It reads config, talks to Valkey, downloads from R2, dispatches to Vast.ai HTTP client, then writes results to Postgres. It imports `src.types.H36Key` (one ML type) and lazily imports ML modules for `detect_video_task`. It belongs in `backend/` alongside the other orchestration code.

**Files:**
- Move: `ml/src/worker.py` → `backend/app/worker.py`
- Modify: `ml/pyproject.toml` (remove skating-backend dep, remove unused deps)
- Modify: `backend/pyproject.toml` (add skating-ml dep for types)

- [ ] **Step 1: Copy ml/src/worker.py to backend/app/worker.py**

Copy the entire file. Fix all imports:

```python
# Line 22: was "from backend.app.config import get_settings"
from app.config import get_settings

# Line 23: was "from backend.app.storage import download_file"
from app.storage import download_file

# Lines 24-31: was "from backend.app.task_manager import ..."
from app.task_manager import (
    TaskStatus,
    get_valkey_client,
    is_cancelled,
    mark_cancelled,
    store_error,
    store_result,
)

# Line 32: keep as-is (will import from ml/ package)
from src.types import H36Key

# Line 210 (lazy): was "from src.vastai.client import process_video_remote_async"
from app.vastai.client import process_video_remote_async

# Lines 208-209 (lazy): was "from backend.app.crud.session import ..." and "from backend.app.database import ..."
from app.crud.session import get_by_id
from app.database import async_session

# Lines 286-288 (lazy): was "from backend.app.crud.session import ...", "from backend.app.database import ...", "from backend.app.services.session_saver import ..."
from app.crud.session import update_session_analysis
from app.database import async_session
from app.services.session_saver import save_analysis_results

# Lines 350-354 (lazy, detect_video_task):
from app.storage import download_file
from src.device import DeviceConfig
from src.pose_estimation.pose_extractor import PoseExtractor
from src.utils.video import get_video_meta
from src.web_helpers import render_person_preview
```

The module docstring should be updated:

```python
"""arq worker for video processing pipeline.

Run with: uv run python -m app.worker

Dispatches all processing to Vast.ai Serverless GPU.
No local GPU fallback.
"""
```

- [ ] **Step 2: Delete old worker.py**

```bash
rm ml/src/worker.py
```

- [ ] **Step 3: Add skating-ml to backend/pyproject.toml**

The worker needs `src.types.H36Key` from ml/. Add the dependency:

```toml
# backend/pyproject.toml dependencies list, add at the end:
    "skating-ml @ file:///home/michael/Github/skating-biomechanics-ml/ml",
```

Note: this brings in ML deps (onnxruntime-gpu, rtmlib, etc.) into backend's dependency tree. Only the worker process actually imports them (via lazy imports in detect_video_task). The FastAPI server process does NOT import them.

- [ ] **Step 4: Remove skating-backend from ml/pyproject.toml**

`ml/` no longer needs the backend package. Remove this line:

```toml
# DELETE this line from ml/pyproject.toml:
    "skating-backend @ file:///home/michael/Github/skating-biomechanics-ml/backend",
```

Also remove deps that were only needed because of the backend dependency. Check which ml-internal code actually uses these:

- `arq` — was used by worker.py (now in backend). Check if any ml/ code still uses it. If not, remove.
- `boto3` — was used by worker.py (now in backend). Check if any ml/ code still uses it. If not, remove.
- `redis` — was used by worker.py (now in backend). Check if any ml/ code still uses it. If not, remove.
- `structlog` — used by ml/ internal code? Check.
- `aiobotocore` — used by ml/gpu_server/server.py. Keep.

To verify, run:

```bash
cd /home/michael/Github/skating-biomechanics-ml && uv run python -c "import ml.src" 2>&1
```

If ml/ imports succeed without these deps, remove them.

Expected removals (verify before committing):
```toml
# These can likely be removed from ml/pyproject.toml:
    "arq>=0.27.0",
    "boto3>=1.42.83",
    "redis>=5.0.0",
    "structlog>=25.5.0",
```

Keep: `aiobotocore>=3.4.0` (used by gpu_server/server.py)

- [ ] **Step 5: Update ml/CLAUDE.md — remove worker references from project structure**

The `ml/CLAUDE.md` project structure lists `worker.py` under `ml/src/`. Remove these lines:

```
│   ├── worker.py                     # arq worker (process_video_task, detect_video_task)
```

And update the Worker Jobs section to point to `backend/app/worker.py`.

- [ ] **Step 6: Update root CLAUDE.md — fix architecture description**

The root CLAUDE.md architecture diagram says:

```
Frontend → FastAPI (backend/) → Valkey queue → arq worker (ml/src/)
```

Change to:

```
Frontend → FastAPI (backend/) → Valkey queue → arq worker (backend/app/worker.py)
```

And the line about Vast.ai:
```
ml depends on backend for infrastructure, never for ML
```

Remove or update any reference to `ml/src/worker.py` in the architecture section.

- [ ] **Step 7: Run backend tests**

Run: `uv run pytest backend/tests/ -v --no-cov`
Expected: All tests PASS (test_vastai_client from Task 1 + existing tests)

Note: `backend/tests/test_task_manager.py` stays in backend/tests/ — it was already importing from `backend.app.task_manager` directly. No change needed.

- [ ] **Step 8: Verify ml/ has zero backend imports**

Run: `grep -rn "from backend" ml/src/ ml/tests/`
Expected: No output (zero matches)

Also verify:
```bash
grep -rn "import backend" ml/src/ ml/tests/
```
Expected: No output

- [ ] **Step 9: Commit**

```bash
git add backend/app/worker.py backend/pyproject.toml ml/pyproject.toml
git rm ml/src/worker.py
git add CLAUDE.md ml/CLAUDE.md backend/CLAUDE.md
git commit -m "refactor: move arq worker from ml/ to backend/, drop ml→backend dependency"
```

---

### Task 3: Clean up orphaned tests and dead code

**Rationale:** After moving worker and vastai client, some ml/ tests still import from backend. Those tests should move to backend/ or be rewritten.

**Files:**
- Move: `ml/tests/test_task_manager.py` → `backend/tests/test_task_manager.py`
- Modify: `ml/tests/ml/test_integration.py` (remove ProcessRequest tests)

- [ ] **Step 1: Move test_task_manager.py to backend/tests/**

The file already imports from `backend.app.task_manager` directly. Just move it:

```bash
git mv ml/tests/test_task_manager.py backend/tests/test_task_manager.py
```

No import changes needed — the imports already use `backend.app.task_manager`.

- [ ] **Step 2: Fix test_integration.py ProcessRequest tests**

`ml/tests/ml/test_integration.py` lines 72-103 import `backend.app.schemas.ProcessRequest`. These 2 tests belong in `backend/tests/`.

Remove the two test methods from `ml/tests/ml/test_integration.py`:

```python
# DELETE these two methods from TestMLPipelineIntegration:
    def test_process_request_schema_accepts_ml_flags(self):
        ...

    def test_process_request_defaults_ml_flags_false(self):
        ...
```

These are schema validation tests for a backend Pydantic model — they have no business in ml/tests/.

- [ ] **Step 3: Verify ml/tests/ has zero backend imports**

Run: `grep -rn "from backend\|import backend" ml/tests/`
Expected: No output

- [ ] **Step 4: Run all tests**

Run: `uv run pytest backend/tests/ ml/tests/ -v --no-cov`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/tests/test_task_manager.py ml/tests/ml/test_integration.py
git rm ml/tests/test_task_manager.py
git commit -m "refactor: move task_manager tests to backend/, remove backend schema tests from ml/"
```

---

### Task 4: Update Taskfile and run commands

**Rationale:** The worker run command currently uses `uv run python -m src.worker`. It must change to `uv run python -m app.worker` (from backend/ context) or be invoked via the backend package.

**Files:**
- Modify: `Taskfile.yml` (worker run command)
- Modify: `CLAUDE.md` (worker references)

- [ ] **Step 1: Find all references to `src.worker` or `ml/src/worker`**

Run: `grep -rn "src.worker\|ml/src/worker\|ml\.src\.worker" --include="*.md" --include="*.yaml" --include="*.yml" --include="*.toml" --include="*.py" --include="*.sh" .`

This will show all references to update. For each match:

- `ml/src/worker.py` → `backend/app/worker.py`
- `python -m src.worker` → `python -m app.worker` (run from backend/ dir)
- `src.worker` → `app.worker`

Do NOT modify files under `data/plans/` or `docs/plans/` — those are historical.

- [ ] **Step 2: Update Taskfile.yml**

If there's a task that starts the worker, update it. For example, if there's:

```yaml
  worker:
    cmds:
      - uv run python -m src.worker
```

Change to:

```yaml
  worker:
    dir: backend
    cmds:
      - uv run python -m app.worker
```

If no such task exists, skip this step.

- [ ] **Step 3: Update ROADMAP.md if it references worker location**

Search ROADMAP.md for "ml/src/worker" and update to "backend/app/worker".

- [ ] **Step 4: Run final verification**

Run: `grep -rn "src\.worker\|ml/src/worker\|ml\.src\.worker" --include="*.md" --include="*.yaml" --include="*.yml" --include="*.toml" --include="*.py" --include="*.sh" . | grep -v "data/plans\|docs/plans"`
Expected: No output (all non-plan references updated)

- [ ] **Step 5: Run all tests one final time**

Run: `uv run pytest backend/tests/ ml/tests/ -v --no-cov`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add Taskfile.yml ROADMAP.md
git commit -m "docs: update worker references from ml/src/ to backend/app/"
```

---

### Task 5: Verify and validate

**Rationale:** Final verification that ml/ is a pure ML library and all orchestrator code lives in backend/.

- [ ] **Step 1: Verify ml/ dependency tree has no backend**

Run: `cd ml && uv pip list 2>/dev/null | grep -i skating`
Expected: Only `skating-ml` appears. No `skating-backend`.

Alternative: `grep skating-backend ml/pyproject.toml`
Expected: No output

- [ ] **Step 2: Verify ml/src/ has zero backend imports**

Run: `grep -rn "from backend\|import backend" ml/src/`
Expected: No output

- [ ] **Step 3: Verify backend/app/worker.py imports from correct locations**

Run: `grep "^from\|^import" backend/app/worker.py`
Expected: Imports from `app.*` (backend internals) and `src.*` (ml types/dispatch only)

- [ ] **Step 4: Verify gpu_server/ unchanged**

Run: `grep -rn "backend\|from backend" ml/gpu_server/`
Expected: No output (gpu server was already independent)

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest backend/tests/ ml/tests/ -v --no-cov`
Expected: All tests PASS

- [ ] **Step 6: Lint check**

Run: `uv run ruff check backend/app/worker.py backend/app/vastai/`
Expected: No errors
