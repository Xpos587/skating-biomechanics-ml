# Choreography Pipeline: Parallelism & Async Design

**Date:** 2026-04-19
**Status:** DRAFT
**Scope:** Parallelizing and async-ifying the choreography planner pipeline
**Related:** `docs/specs/2026-04-12-choreography-planner-design.md`

---

## Problem Statement

The choreography planner has three categories of unnecessary serialization:

1. **Music analysis blocks the HTTP response.** `upload_music` runs librosa/madmom/MSAF (5-15s) and R2 upload inside the request handler before returning. The user stares at a spinner for the entire duration.

2. **Analysis steps run serially.** Four independent analysis stages (BPM, energy, peaks, MSAF) execute sequentially. BPM + energy + peaks all operate on the same `librosa.load()` output and can run concurrently. MSAF is fully independent (takes a file path, loads audio itself).

3. **Frontend waits on server for pure-client work.** The inventory editor has zero dependency on music analysis but is hidden until it completes. The rink SVG renderer is pure string concatenation with no server dependencies, yet every layout switch requires a network round-trip. The CSP solver (1-5ms) runs synchronously inside an `async def`, blocking the event loop.

Additionally, two bugs were discovered during research (see Appendix).

---

## Target Architecture

```
POST /music/upload
    |
    +---> save tempfile, create DB record (status="analyzing")
    +---> return {music_id} immediately
    |
    +---> Background (asyncio.gather):
    |       Track A: ThreadPoolExecutor(
    |           librosa.load()         -- single load, shared
    |           BPM  |  energy+peaks   -- concurrent after load
    |           MSAF                  -- independent, runs in parallel
    |       )
    |       Track B: R2 upload        -- I/O, runs in parallel
    |
    +---> update DB with results, status="completed"


POST /generate
    |
    +---> asyncio.to_thread(solve_layout)  -- unblock event loop
    +---> return {layouts}

Frontend (on generate success):
    +---> Promise.all(render_rink for each layout)  -- prefetch all SVGs
    +---> layout switching = instant (client-side state)


GET /music/{id}/analysis  or  SSE /music/{id}/progress
    |
    +---> poll status: analyzing | completed | failed
```

---

## Recommendations by Layer

### Frontend

| Change | What | Feasibility | Effort | Impact |
|--------|------|-------------|--------|--------|
| Show inventory immediately | Remove gating on music analysis status. The inventory editor is pure client state. | Easy | 30 min | Users work during 3-15s wait |
| Client-side rink renderer | Port `rink_renderer.py` to `rink-renderer.ts`. The function is string concatenation with zero server deps. | Easy | 2-3 hours | Eliminates network hop on every layout switch |
| Prefetch all rink SVGs | On generate success, `Promise.all` to render all 3 layouts client-side. | Easy | 30 min | Zero-latency layout switching |
| Optimistic delete/save | React Query `onMutate` for program delete and title save. | Easy | 1 hour | Instant UI feedback |
| Client-side PDF export | jsPDF + svg2pdf.js. Removes headless browser dependency. | Medium | 3-4 hours | No server-side rendering dep |
| SSE for analysis progress | Reuse existing `sse-starlette` + `task_manager` to stream analysis status. | Medium | 2-3 hours | Real-time progress feedback |

### API

| Change | What | Feasibility | Effort | Impact |
|--------|------|-------------|--------|--------|
| Immediate response on upload | Return `{music_id}` before analysis starts. Frontend polls `GET /music/{id}/analysis` for status. | Easy | 1 hour | User gets instant feedback |
| `asyncio.to_thread(solve_layout)` | One-line fix in `generate_layout`. CSP solver is sync, blocking the event loop. | Easy | 5 min | Non-blocking event loop |
| SSE endpoint for progress | `GET /choreography/music/{id}/progress` using existing SSE infrastructure. | Easy | 1-2 hours | Real-time analysis progress |

### Backend Services

| Change | What | Feasibility | Effort | Impact |
|--------|------|-------------|--------|--------|
| Parallel R2 upload + analysis | `asyncio.gather` around `analyze_music_sync` and `upload_file` in `upload_music`. | Easy | 15 min | Wall-clock reduction for upload path |
| Step parallelism within analysis | After `librosa.load()`, run BPM, energy+peaks concurrently via `ThreadPoolExecutor.submit`. MSAF runs independently in its own thread. All release GIL (NumPy/Cython). | Medium | 1-2 hours | 30-50% wall-clock reduction for analysis |
| Migrate to OR-Tools CP-SAT | Replace random search (500 iterations, 5x10^-8 coverage) with CP-SAT solver. Provably optimal results, ~100ms solve time. Supports `num_search_workers` for genuine parallelism. | Medium | 2-3 days | Quality: provably optimal layouts. Speed: 500 random iterations -> deterministic solve. |
| Audio fingerprint caching | Chromaprint fingerprint on upload. Deduplicate same competition music across athletes (5-10x re-upload rate). | Medium | 3-4 hours | Eliminates redundant analysis for duplicate music |

### Worker

| Change | What | Feasibility | Effort | Impact |
|--------|------|-------------|--------|--------|
| Music analysis as arq job | Move from route handler to arq worker. 5-15s, 300MB RAM, CPU-heavy. Needs timeout=120s, max_tries=1 (failures are deterministic). | Medium | 2-3 hours | Respects "zero ML imports in backend" constraint. Worker handles heavy CPU without blocking API. |
| Keep CSP/render/validate synchronous | These are sub-millisecond pure Python. No benefit from arq overhead. | N/A | 0 min | Correct placement already |

**Task decomposition summary:**

| Task | arq job? | Duration | Why |
|------|----------|----------|-----|
| Music analysis | YES | 5-15s | CPU-heavy, needs progress, respects backend constraint |
| CSP generation | NO | <1s | Pure Python, trivial RAM |
| Rink rendering | NO | <10ms | String concatenation |
| Validation | NO | <1ms | Pure function |
| PDF export | YES (future) | 5-30s | Headless browser |

---

## Migration Path

### Phase 0: Bug fixes (prerequisites)

- Add `setuptools` to backend deps (fixes `pkg_resources` import error on Python 3.13, breaks madmom import)
- Remove `_priority` parameter from `enqueue_job` calls in `detect.py` and `process.py` (silently ignored by arq)

### Phase 1: Quick wins (no architecture change)

1. `asyncio.to_thread(solve_layout)` in `generate_layout` -- 5 min
2. `asyncio.gather(analysis, R2_upload)` in `upload_music` -- 15 min
3. Show inventory immediately in frontend -- 30 min
4. Immediate response pattern: return `music_id` before analysis -- 1 hour
5. Prefetch rink SVGs on generate success -- 30 min

### Phase 2: Frontend rendering (highest UX impact)

1. Port `rink_renderer.py` to `rink-renderer.ts` -- 2-3 hours
2. Optimistic delete/save with React Query -- 1 hour
3. SSE endpoint for analysis progress -- 1-2 hours

### Phase 3: Analysis parallelism + worker migration

1. Step parallelism within `_run_analysis` (concurrent BPM, energy, peaks) -- 1-2 hours
2. Music analysis as arq job -- 2-3 hours
3. Audio fingerprint caching -- 3-4 hours

### Phase 4: Quality improvement

1. Migrate CSP solver to OR-Tools CP-SAT -- 2-3 days
2. Client-side PDF export -- 3-4 hours

---

## Bugs Found

### Bug 1: `_priority` parameter silently ignored by arq

`enqueue_job` calls in `backend/app/routes/detect.py` and `backend/app/routes/process.py` pass `_priority=<int>`. The arq library accepts `**kwargs` but silently ignores unknown parameters -- priority is not implemented in arq's enqueue API. These calls have no effect.

**Fix:** Remove `_priority` from all `enqueue_job` calls, or document as a no-op.

### Bug 2: `pkg_resources` import breaks on Python 3.13

madmom (dependency of `music_analyzer.py`) imports `pkg_resources` internally. Python 3.13 removed `pkg_resources` from the standard library, causing `ImportError` at runtime. The docstring in `music_analyzer.py` says "called from arq worker" but the route handler imports and calls `analyze_music_sync` directly.

**Fix:** Add `setuptools` to `backend/pyproject.toml` (provides `pkg_resources` shim). Alternatively, move music analysis to the ML worker process where `setuptools` is already available.

---

## Appendix: Why Not ProcessPoolExecutor for CSP

The CSP solver's 500-iteration random search could theoretically run in parallel via `ProcessPoolExecutor` (3-4x speedup). This was rejected because:

1. The solver takes 1-5ms total. Process spawn overhead is ~50ms. Parallelism would make it slower.
2. The real problem is quality (5x10^-8 search space coverage), not speed. OR-Tools CP-SAT solves both: provably optimal in ~100ms.
3. `num_search_workers` in OR-Tools CP-SAT provides genuine parallelism if needed in the future.
