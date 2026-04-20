# Choreography Pipeline Parallelism — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parallelize and async-ify the choreography pipeline — eliminate blocking HTTP responses, parallelize analysis steps, move rendering client-side.

**Architecture:** Immediate response on music upload (analysis runs in background). Parallel step execution within analysis (BPM/energy/MSAF via ThreadPoolExecutor). Client-side rink SVG rendering eliminates server round-trip. CSP solver wrapped in `asyncio.to_thread`.

**Tech Stack:** FastAPI + asyncio + concurrent.futures, React Query, Zod, arq

**Spec:** `docs/specs/2026-04-19-choreography-parallelism-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `backend/app/routes/detect.py:59` | MODIFY | Remove `_priority` no-op |
| `backend/app/routes/process.py:68` | MODIFY | Remove `_priority` no-op |
| `backend/app/services/choreography/music_analyzer.py` | MODIFY | Parallel step execution |
| `backend/app/routes/choreography.py` | MODIFY | Immediate response + parallel gather + to_thread for CSP |
| `frontend/src/lib/rink-renderer.ts` | CREATE | Client-side SVG rink renderer |
| `frontend/src/components/choreography/rink-diagram.tsx` | MODIFY | Use client-side renderer instead of server API |
| `frontend/src/lib/api/choreography.ts` | MODIFY | Remove `useRenderRink` mutation, add optimistic delete |
| `frontend/src/app/(app)/choreography/new/page.tsx` | MODIFY | Show inventory immediately, client-side rink rendering |
| `frontend/src/components/choreography/layout-picker.tsx` | UNCHANGED | |
| `frontend/src/app/(app)/choreography/page.tsx` | MODIFY | Optimistic program delete |

---

## Task 1: Remove `_priority` no-op from arq enqueue calls

**Files:**
- Modify: `backend/app/routes/detect.py:59`
- Modify: `backend/app/routes/process.py:68`
- Test: `backend/tests/` (existing tests must still pass)

- [ ] **Step 1: Remove `_priority` from detect.py**

In `backend/app/routes/detect.py`, line 59, remove `_priority=0,` from the `enqueue_job` call. Keep all other kwargs unchanged.

- [ ] **Step 2: Remove `_priority` from process.py**

In `backend/app/routes/process.py`, line 68, remove `_priority=10,` from the `enqueue_job` call. Keep all other kwargs unchanged.

- [ ] **Step 3: Run backend tests**

Run: `cd /home/michael/Github/skating-biomechanics-ml && uv run pytest backend/tests/ -x -q`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add backend/app/routes/detect.py backend/app/routes/process.py
git commit -m "fix(arq): remove _priority no-op from enqueue_job calls"
```

---

## Task 2: Wrap CSP solver in `asyncio.to_thread`

**Files:**
- Modify: `backend/app/routes/choreography.py:212`

- [ ] **Step 1: Wrap solve_layout call**

In `backend/app/routes/choreography.py`, in the `generate_layout` endpoint (line ~212), replace:

```python
    layouts = solve_layout(
        inventory=body.inventory,
        music_features=music_features,
        discipline=body.discipline,
        segment=body.segment,
    )
```

with:

```python
    import asyncio

    layouts = await asyncio.to_thread(
        solve_layout,
        inventory=body.inventory,
        music_features=music_features,
        discipline=body.discipline,
        segment=body.segment,
    )
```

Remove the `import asyncio` line that was already at the top of the file (inside the `upload_music` function) — it should become a module-level import if not already. Check: if `asyncio` is already imported at module level, just use it. If it's only imported inside `upload_music`, move it to the top of the file.

- [ ] **Step 2: Run backend tests**

Run: `cd /home/michael/Github/skating-biomechanics-ml && uv run pytest backend/tests/ -x -q`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add backend/app/routes/choreography.py
git commit -m "refactor(choreography): wrap solve_layout in asyncio.to_thread"
```

---

## Task 3: Parallel music analysis steps

**Files:**
- Modify: `backend/app/services/choreography/music_analyzer.py`
- Test: `backend/tests/services/choreography/test_music_analyzer.py` (may need to create)

- [ ] **Step 1: Write failing test for parallel analysis**

Create `backend/tests/services/choreography/test_music_analyzer.py`:

```python
"""Tests for parallel music analysis."""

from unittest.mock import patch

import pytest


def test_analyze_music_sync_returns_expected_keys():
    """analyze_music_sync returns all required keys."""
    from app.services.choreography.music_analyzer import analyze_music_sync

    # Mock all heavy dependencies
    with (
        patch("app.services.choreography.music_analyzer._run_analysis") as mock_run,
    ):
        mock_run.return_value = {
            "bpm": 120.0,
            "duration_sec": 180.0,
            "peaks": [30.0, 60.0],
            "structure": [{"type": "verse", "start": 0.0, "end": 30.0}],
            "energy_curve": {"timestamps": [0.0, 0.5], "values": [0.1, 0.2]},
        }
        result = analyze_music_sync("/fake/path.mp3")

    assert "bpm" in result
    assert "duration_sec" in result
    assert "peaks" in result
    assert "structure" in result
    assert "energy_curve" in result


def test_extract_features_for_csp():
    """extract_features_for_csp extracts only CSP-relevant fields."""
    from app.services.choreography.music_analyzer import extract_features_for_csp

    full = {
        "bpm": 120.0,
        "duration_sec": 180.0,
        "peaks": [30.0, 60.0],
        "structure": [{"type": "verse", "start": 0.0, "end": 30.0}],
        "energy_curve": {"timestamps": [0.0, 0.5], "values": [0.1, 0.2]},
    }
    features = extract_features_for_csp(full)

    assert features["duration"] == 180.0
    assert features["peaks"] == [30.0, 60.0]
    assert features["structure"] == [{"type": "verse", "start": 0.0, "end": 30.0}]
    assert "bpm" not in features
    assert "energy_curve" not in features
```

- [ ] **Step 2: Run test to verify it passes**

Run: `cd /home/michael/Github/skating-biomechanics-ml && uv run pytest backend/tests/services/choreography/test_music_analyzer.py -v`
Expected: PASS (tests mock `_run_analysis`, so no ML deps needed)

- [ ] **Step 3: Refactor `_run_analysis` for parallel execution**

Replace the body of `_run_analysis` in `backend/app/services/choreography/music_analyzer.py` with a parallel version. The new implementation:

```python
def _run_analysis(audio_path: str) -> dict:
    """Run music analysis pipeline with parallel step execution.

    Steps:
    1. librosa.load() — single load, shared across steps
    2. Concurrent: BPM (madmom/librosa), energy+peaks, MSAF structure
    """
    import librosa
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor, as_completed

    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    duration_sec = float(len(y) / sr)

    def _compute_bpm(y: np.ndarray, sr: int) -> float:
        """BPM via madmom with librosa fallback."""
        bpm = None
        try:
            from madmom.features.beats import DBNBeatTracker

            act = DBNBeatTracker.preprocess(y, sr=sr)
            beat_frames = DBNBeatTracker.detect(act, fps=sr / 512)
            if len(beat_frames) > 1:
                intervals = np.diff(beat_frames) * 512 / sr
                bpm = float(60.0 / np.median(intervals))
        except Exception:  # noqa: BLE001
            logger.warning("madmom beat tracking failed, using librosa fallback")

        if bpm is None:
            bpm = float(librosa.beat.beat_track(y=y, sr=sr)[0])
        return round(bpm, 1)

    def _compute_energy_peaks(y: np.ndarray, sr: int) -> tuple[list[float], list[float]]:
        """Energy curve + peak detection."""
        import librosa

        hop_length = int(sr * 0.5)
        rms = librosa.feature.rms(y=y, frame_length=hop_length * 2, hop_length=hop_length)[0]
        timestamps = [float(i * 0.5) for i in range(len(rms))]
        energy_curve = {"timestamps": timestamps, "values": [float(v) for v in rms]}

        peaks: list[float] = []
        try:
            from scipy.signal import find_peaks

            rms_normalized = (rms - rms.min()) / (rms.max() - rms.min() + 1e-8)
            peak_indices, _ = find_peaks(rms_normalized, height=0.6, distance=4)
            peaks = [timestamps[i] for i in peak_indices]
        except Exception:  # noqa: BLE001
            logger.warning("Peak detection failed")

        return peaks, energy_curve

    def _compute_structure(audio_path: str) -> list[dict]:
        """Structure boundaries via MSAF."""
        structure: list[dict] = []
        try:
            import msaf

            boundaries, labels = msaf.process(audio_path, boundaries_id="sf", labels_id="foote")
            for i in range(len(boundaries) - 1):
                structure.append(
                    {
                        "type": labels[i] if i < len(labels) else "unknown",
                        "start": float(boundaries[i]),
                        "end": float(boundaries[i + 1]),
                    }
                )
        except Exception:  # noqa: BLE001
            logger.warning("MSAF structure analysis failed -- using empty structure")
        return structure

    # Run all three steps concurrently
    with ThreadPoolExecutor(max_workers=3) as executor:
        bpm_future = executor.submit(_compute_bpm, y, sr)
        energy_future = executor.submit(_compute_energy_peaks, y, sr)
        structure_future = executor.submit(_compute_structure, audio_path)

        bpm = bpm_future.result()
        peaks, energy_curve = energy_future.result()
        structure = structure_future.result()

    return {
        "bpm": bpm,
        "duration_sec": round(duration_sec, 1),
        "peaks": peaks,
        "structure": structure,
        "energy_curve": energy_curve,
    }
```

- [ ] **Step 4: Run tests**

Run: `cd /home/michael/Github/skating-biomechanics-ml && uv run pytest backend/tests/services/choreography/test_music_analyzer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/choreography/music_analyzer.py backend/tests/services/choreography/test_music_analyzer.py
git commit -m "perf(choreography): parallel music analysis steps via ThreadPoolExecutor"
```

---

## Task 4: Parallel R2 upload + analysis in upload route

**Files:**
- Modify: `backend/app/routes/choreography.py:73-154`

- [ ] **Step 1: Refactor upload_music to use asyncio.gather**

In `backend/app/routes/choreography.py`, replace the `try` block inside `upload_music` (lines ~102-143) with a version that runs analysis and R2 upload concurrently:

```python
    try:
        import asyncio
        import logging

        logger = logging.getLogger(__name__)

        duration_sec = 0.0
        bpm = None
        energy_curve = None
        peaks = None
        structure = None

        r2_key = f"music/{user.id}/{music.id}{suffix}"

        async def _analyze():
            nonlocal duration_sec, bpm, energy_curve, peaks, structure
            try:
                from app.services.choreography.music_analyzer import analyze_music_sync

                logger.info("Running music analysis on %s", tmp_path)
                result = await asyncio.to_thread(analyze_music_sync, tmp_path)
                duration_sec = result["duration_sec"]
                bpm = result["bpm"]
                energy_curve = result["energy_curve"]
                peaks = result["peaks"]
                structure = result.get("structure") or []
                logger.info("Analysis complete: bpm=%.1f, duration=%.1f", bpm, duration_sec)
            except ImportError:
                logger.info("librosa not available, using basic duration estimation")
                duration_sec = await asyncio.to_thread(_get_duration, tmp_path, suffix)

        async def _upload():
            logger.info("Uploading to R2: %s", r2_key)
            await asyncio.to_thread(upload_file, tmp_path, r2_key)
            logger.info("R2 upload complete")

        await asyncio.gather(_analyze(), _upload())

        await update_music_analysis(
            db,
            music,
            audio_url=f"/files/{r2_key}",
            duration_sec=duration_sec,
            bpm=bpm,
            energy_curve=energy_curve,
            peaks=peaks,
            structure=structure,
            status="completed",
        )
    except Exception as e:
        # ... existing error handling unchanged ...
```

The key changes:
1. Move `import asyncio` and `import logging` to the top of the `try` block (or module level)
2. Compute `r2_key` before the parallel section
3. Define `_analyze()` and `_upload()` as inner coroutines
4. `asyncio.gather(_analyze(), _upload())` replaces the sequential calls
5. Remove the separate `await asyncio.to_thread(upload_file, ...)` call
6. Keep the existing `except` and `finally` blocks unchanged

- [ ] **Step 2: Run backend tests**

Run: `cd /home/michael/Github/skating-biomechanics-ml && uv run pytest backend/tests/ -x -q`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add backend/app/routes/choreography.py
git commit -m "perf(choreography): parallel R2 upload and music analysis via asyncio.gather"
```

---

## Task 5: Client-side rink SVG renderer

**Files:**
- Create: `frontend/src/lib/rink-renderer.ts`

- [ ] **Step 1: Create `rink-renderer.ts`**

Port `backend/app/services/choreography/rink_renderer.py` to TypeScript. Create `frontend/src/lib/rink-renderer.ts`:

```ts
/**
 * Renders a top-down orthographic view of a 60m x 30m ice rink
 * with element markers, labels, and connecting paths.
 */

interface RinkElement {
  code: string
  position?: { x: number; y: number }
  timestamp?: number
}

const RINK_W = 60
const RINK_H = 30

function isSpin(code: string): boolean {
  return code.includes("Sp")
}

function isStep(code: string): boolean {
  return code.includes("StSq")
}

function isChoreo(code: string): boolean {
  return code.includes("ChSq")
}

function elementColor(code: string): string {
  if (isSpin(code)) return "#9333ea"
  if (isStep(code)) return "#16a34a"
  if (isChoreo(code)) return "#2563eb"
  return "#ea580c"
}

function elementShape(el: RinkElement, x: number, y: number): string {
  const color = elementColor(el.code)

  if (isSpin(el.code)) {
    return `<circle cx="${x}" cy="${y}" r="1.2" fill="${color}" opacity="0.3" stroke="${color}" stroke-width="0.1"/>`
  }
  if (isStep(el.code)) {
    return `<rect x="${x - 1}" y="${y - 0.5}" width="2" height="1" fill="none" stroke="${color}" stroke-width="0.1" stroke-dasharray="0.3,0.2"/>`
  }
  if (isChoreo(el.code)) {
    return `<polygon points="${x},${y - 0.8} ${x + 0.8},${y} ${x},${y + 0.8} ${x - 0.8},${y}" fill="${color}" opacity="0.3" stroke="${color}" stroke-width="0.1"/>`
  }
  return `<circle cx="${x}" cy="${y}" r="0.6" fill="${color}" opacity="0.8"/>`
}

export function renderRink(
  elements: RinkElement[],
  options?: { width?: number; height?: number },
): string {
  const { width = 1200, height = 600 } = options ?? {}

  const parts: string[] = []

  // Rink outline
  parts.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 60 30">`)
  parts.push(`<rect x="0" y="0" width="60" height="30" fill="#e8f0fe" rx="1"/>`)
  parts.push(`<rect x="1" y="1" width="58" height="28" fill="none" stroke="#2563eb" stroke-width="0.15" rx="0.5"/>`)
  parts.push(`<line x1="30" y1="1" x2="30" y2="29" stroke="#dc2626" stroke-width="0.1" stroke-dasharray="0.5,0.5"/>`)
  parts.push(`<circle cx="30" cy="15" r="4.5" fill="none" stroke="#dc2626" stroke-width="0.1"/>`)
  parts.push(`<circle cx="30" cy="15" r="0.15" fill="#dc2626"/>`)
  parts.push(`<line x1="5" y1="1" x2="5" y2="29" stroke="#2563eb" stroke-width="0.08"/>`)
  parts.push(`<line x1="55" y1="1" x2="55" y2="29" stroke="#2563eb" stroke-width="0.08"/>`)

  // Faceoff circles
  for (const [cx, cy] of [[10, 7.5], [10, 22.5], [50, 7.5], [50, 22.5]]) {
    parts.push(`<circle cx="${cx}" cy="${cy}" r="3" fill="none" stroke="#2563eb" stroke-width="0.08"/>`)
    parts.push(`<circle cx="${cx}" cy="${cy}" r="0.15" fill="#dc2626"/>`)
  }

  // Elements
  for (let i = 0; i < elements.length; i++) {
    const el = elements[i]
    const pos = el.position
    if (!pos) continue

    const x = pos.x
    const y = pos.y
    const color = elementColor(el.code)

    parts.push(elementShape(el, x, y))
    parts.push(`<text x="${x}" y="${y - 1.2}" text-anchor="middle" font-size="1.2" fill="${color}" font-weight="bold">${el.code}</text>`)
    parts.push(`<text x="${x}" y="${y + 0.3}" text-anchor="middle" font-size="0.7" fill="#666">${i + 1}</text>`)

    // Connecting path to next element
    if (i < elements.length - 1) {
      const nextPos = elements[i + 1].position
      if (nextPos) {
        parts.push(`<line x1="${x}" y1="${y}" x2="${nextPos.x}" y2="${nextPos.y}" stroke="#94a3b8" stroke-width="0.06" stroke-dasharray="0.3,0.2" opacity="0.6"/>`)
      }
    }
  }

  parts.push("</svg>")
  return parts.join("\n")
}
```

- [ ] **Step 2: Write test for rink renderer**

Create `frontend/src/lib/__tests__/rink-renderer.test.ts`:

```ts
import { describe, expect, it } from "vitest"
import { renderRink } from "../rink-renderer"

describe("renderRink", () => {
  it("returns SVG string with rink outline", () => {
    const svg = renderRink([])
    expect(svg).toContain("<svg")
    expect(svg).toContain("</svg>")
    expect(svg).toContain('width="1200"')
    expect(svg).toContain('viewBox="0 0 60 30"')
  })

  it("renders jump elements", () => {
    const svg = renderRink([{ code: "3Lz", position: { x: 30, y: 15 } }])
    expect(svg).toContain("3Lz")
    expect(svg).toContain("#ea580c")
  })

  it("renders spin elements", () => {
    const svg = renderRink([{ code: "CSp4", position: { x: 20, y: 10 } }])
    expect(svg).toContain("CSp4")
    expect(svg).toContain("#9333ea")
  })

  it("renders step sequences", () => {
    const svg = renderRink([{ code: "StSq4", position: { x: 40, y: 10 } }])
    expect(svg).toContain("StSq4")
    expect(svg).toContain("#16a34a")
  })

  it("draws connecting paths between elements", () => {
    const svg = renderRink([
      { code: "3Lz", position: { x: 20, y: 15 } },
      { code: "3F", position: { x: 40, y: 15 } },
    ])
    expect(svg).toContain("x1=\"20\"")
    expect(svg).toContain("x2=\"40\"")
  })

  it("skips elements without position", () => {
    const svg = renderRink([{ code: "3Lz" }])
    expect(svg).not.toContain("3Lz")
  })

  it("supports custom width/height", () => {
    const svg = renderRink([], { width: 600, height: 300 })
    expect(svg).toContain('width="600"')
    expect(svg).toContain('height="300"')
  })
})
```

- [ ] **Step 3: Run test**

Run: `cd /home/michael/Github/skating-biomechanics-ml/frontend && bunx vitest run src/lib/__tests__/rink-renderer.test.ts`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add frontend/src/lib/rink-renderer.ts frontend/src/lib/__tests__/rink-renderer.test.ts
git commit -m "feat(choreography): client-side rink SVG renderer"
```

---

## Task 6: Switch frontend to client-side rink rendering

**Files:**
- Modify: `frontend/src/components/choreography/rink-diagram.tsx`
- Modify: `frontend/src/lib/api/choreography.ts` (remove `useRenderRink`)
- Modify: `frontend/src/app/(app)/choreography/new/page.tsx`

- [ ] **Step 1: Rewrite RinkDiagram to use client-side renderer**

Replace the entire content of `frontend/src/components/choreography/rink-diagram.tsx`:

```tsx
"use client"

import { useMemo } from "react"
import { renderRink } from "@/lib/rink-renderer"
import type { LayoutElement } from "@/types/choreography"

interface RinkDiagramProps {
  elements: LayoutElement[]
  className?: string
}

export function RinkDiagram({ elements, className }: RinkDiagramProps) {
  const svgHtml = useMemo(() => {
    const rinkElements = elements
      .filter(el => el.position)
      .map(el => ({
        code: el.code,
        position: el.position,
      }))
    if (rinkElements.length === 0) return null
    return renderRink(rinkElements)
  }, [elements])

  if (!svgHtml) {
    return (
      <div className={`flex items-center justify-center rounded-2xl border border-dashed border-border p-8 text-sm text-muted-foreground ${className ?? ""}`}>
        No elements with positions
      </div>
    )
  }

  return (
    <div
      className={`rounded-2xl border border-border p-2 ${className ?? ""}`}
      style={{ backgroundColor: "oklch(var(--background))" }}
      // biome-ignore lint/security/noDangerouslySetInnerHtml: SVG from local renderer
      dangerouslySetInnerHTML={{ __html: svgHtml }}
    />
  )
}
```

Key changes:
- Accept `elements: LayoutElement[]` instead of `svgHtml: string | null`
- Use `useMemo` to compute SVG from elements (memoized, instant re-render)
- Remove loading state (no network call)

- [ ] **Step 2: Remove `useRenderRink` from API hooks**

In `frontend/src/lib/api/choreography.ts`:

1. Remove the `RenderRinkResponseSchema` Zod schema (the `image_url` schema).
2. Remove the `useRenderRink` function entirely.

- [ ] **Step 3: Update new/page.tsx to use client-side rink**

In `frontend/src/app/(app)/choreography/new/page.tsx`:

1. Remove `useRenderRink` from the import statement (line with `useRenderRink`).
2. Remove the `const renderRink = useRenderRink()` line.
3. In the `LayoutPicker` `onSelect` callback, remove the `renderRink.mutate(...)` call. The callback should only call `setSelectedLayout(...)`.

Replace the `onSelect` handler:

```tsx
onSelect={idx => {
  setSelectedLayout(generateLayouts.data.layouts[idx])
}}
```

4. In the `<RinkDiagram>` usage, replace:

```tsx
<RinkDiagram
  svgHtml={renderRink.data?.image_url ?? null}
  isLoading={renderRink.isPending}
/>
```

with:

```tsx
<RinkDiagram elements={selectedLayout.elements} />
```

- [ ] **Step 4: Run frontend type check and lint**

Run: `cd /home/michael/Github/skating-biomechanics-ml/frontend && bunx tsc --noEmit && bunx biome check src/`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/choreography/rink-diagram.tsx frontend/src/lib/api/choreography.ts frontend/src/app/(app)/choreography/new/page.tsx
git commit -m "feat(choreography): switch to client-side rink SVG rendering"
```

---

## Task 7: Show inventory editor immediately (remove music gating)

**Files:**
- Modify: `frontend/src/app/(app)/choreography/new/page.tsx`

- [ ] **Step 1: Remove `musicReady` gate from inventory section**

In `frontend/src/app/(app)/choreography/new/page.tsx`, find the inventory section (Step 2) and the generate button (Step 3). Both are wrapped in `{musicReady && (...)}`. Change them to always render, but keep the generate button disabled when music is not ready.

Find:

```tsx
      {/* Step 2: Element inventory */}
      {musicReady && (
        <section>
```

Replace with:

```tsx
      {/* Step 2: Element inventory */}
      <section>
```

And remove the closing `)}`.

Do the same for Step 3 (generate button). Find:

```tsx
      {/* Step 3: Generate */}
      {musicReady && (
        <section>
```

Replace with:

```tsx
      {/* Step 3: Generate */}
      <section>
```

And remove the closing `)}`.

The generate button's `disabled` prop already uses `canGenerate` which includes the `musicReady` check, so it will stay disabled until music analysis completes.

- [ ] **Step 2: Run frontend type check**

Run: `cd /home/michael/Github/skating-biomechanics-ml/frontend && bunx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/app/(app)/choreography/new/page.tsx
git commit -m "feat(choreography): show inventory editor immediately, don't wait for music analysis"
```

---

## Task 8: Optimistic program delete

**Files:**
- Modify: `frontend/src/lib/api/choreography.ts`

- [ ] **Step 1: Add optimistic delete to useDeleteProgram**

In `frontend/src/lib/api/choreography.ts`, replace the `useDeleteProgram` function:

```ts
export function useDeleteProgram() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => apiDelete(`/choreography/programs/${id}`),
    onMutate: async (id) => {
      await qc.cancelQueries({ queryKey: ["programs"] })
      const previous = qc.getQueryData(["programs"])
      qc.setQueryData(["programs"], (old: z.infer<typeof ProgramListResponseSchema> | undefined) => {
        if (!old) return old
        return {
          ...old,
          programs: old.programs.filter(p => p.id !== id),
          total: old.total - 1,
        }
      })
      return { previous }
    },
    onError: (_err, _id, context) => {
      if (context?.previous) {
        qc.setQueryData(["programs"], context.previous)
      }
    },
    onSettled: () => {
      qc.invalidateQueries({ queryKey: ["programs"] })
    },
  })
}
```

- [ ] **Step 2: Run frontend type check**

Run: `cd /home/michael/Github/skating-biomechanics-ml/frontend && bunx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/lib/api/choreography.ts
git commit -m "feat(choreography): optimistic UI for program delete"
```

---

## Task 9: Prefetch rink SVGs on generate success

**Files:**
- Modify: `frontend/src/app/(app)/choreography/new/page.tsx`

> Note: With Task 6 complete, rink rendering is client-side and instant. This task is now a no-op — rendering happens in `useMemo` with zero network calls. The `RinkDiagram` component already receives `elements` directly and renders synchronously.

- [ ] **Step 1: Verify no changes needed**

After Task 6, the `RinkDiagram` component takes `elements` as a prop and renders via `useMemo`. There is no network call to prefetch. Layout switching is already instant.

Confirm by checking that `new/page.tsx` passes `elements={selectedLayout.elements}` to `<RinkDiagram>`.

- [ ] **Step 2: Skip commit (no changes)**

Mark this task as complete — the client-side rendering from Task 6 already achieves the goal.

---

## Task 10: Update choreography programs page for optimistic delete

**Files:**
- Modify: `frontend/src/app/(app)/choreography/page.tsx`

- [ ] **Step 1: Verify programs page uses useDeleteProgram**

Read `frontend/src/app/(app)/choreography/page.tsx` and confirm it uses `useDeleteProgram` from `@/lib/api/choreography.ts`. The optimistic behavior from Task 8 applies automatically via the React Query hook — no changes needed in the page component.

If the page has custom delete handling (e.g., `window.confirm`), ensure it calls `deleteProgram.mutate(id)` which already has optimistic updates.

- [ ] **Step 2: Run frontend type check**

Run: `cd /home/michael/Github/skating-biomechanics-ml/frontend && bunx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Skip commit if no changes needed**

If `page.tsx` already uses the hook correctly, no commit needed.

---

## Task 11: Full integration test

**Files:**
- No new files

- [ ] **Step 1: Run full backend test suite**

Run: `cd /home/michael/Github/skating-biomechanics-ml && uv run pytest backend/tests/ -x -q`
Expected: All tests pass

- [ ] **Step 2: Run full frontend type check**

Run: `cd /home/michael/Github/skating-biomechanics-ml/frontend && bunx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Run frontend lint**

Run: `cd /home/michael/Github/skating-biomechanics-ml/frontend && bunx biome check src/`
Expected: No errors

- [ ] **Step 4: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "fix(choreography): integration test fixes for parallelism changes"
```
