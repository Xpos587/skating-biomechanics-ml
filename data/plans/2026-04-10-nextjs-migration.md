# Next.js Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- []`) syntax for tracking.

**Goal:** Migrate the frontend from Vite + React Router to Next.js App Router while keeping Tailwind CSS v4 + shadcn/ui unchanged.

**Architecture:** Next.js App Router with file-based routing. Two client pages (upload, analyze) wrapped in `"use client"` since they use React hooks heavily. API proxy to FastAPI backend via Next.js rewrites (already configured). React Query provider in a shared layout. No SSR needed — all pages are client-interactive.

**Tech Stack:** Next.js 16, React 19, Tailwind CSS v4, shadcn/ui, React Query, Zod, MSW (tests)

---

## File Structure (Target)

```
src/frontend/
├── next.config.ts              # Already exists — proxy rewrite to FastAPI
├── postcss.config.mjs          # Already exists — Tailwind v4
├── tsconfig.json               # Update: add Next.js required fields
├── package.json                # Update: scripts, remove Vite deps, add vitest config
├── src/
│   ├── app/                    # NEW — Next.js App Router
│   │   ├── layout.tsx          # Root layout (QueryClientProvider, header, <html>)
│   │   ├── page.tsx            # "/" → UploadPage (client component)
│   │   ├── analyze/
│   │   │   └── page.tsx        # "/analyze" → AnalyzePage (client component)
│   │   └── globals.css         # Rename of index.css
│   ├── components/ui/          # UNCHANGED — shadcn components
│   ├── lib/                    # UNCHANGED — api.ts, schemas.ts, utils.ts
│   ├── types/                  # UNCHANGED — index.ts
│   └── test/                   # UPDATE — adapt for Next.js
│       ├── setup.ts            # Keep vitest setup (not affected by Next.js)
│       ├── handlers.ts         # UNCHANGED
│       ├── server.ts           # UNCHANGED
│       └── test-utils.tsx      # Update: remove react-router wrappers if any
├── index.html                  # DELETE — Next.js generates HTML
└── vite.config.ts              # DELETE — replaced by next.config.ts
```

## What Changes vs What Stays

### UNCHANGED (no modifications needed)
- `src/components/ui/*` — all 9 shadcn components (already have `"use client"` where needed)
- `src/lib/api.ts` — fetch calls to `/api/*` work identically (proxied by Next.js rewrite)
- `src/lib/schemas.ts` — Zod schemas
- `src/lib/utils.ts` — cn() utility
- `src/lib/useMountEffect.ts` — custom hook
- `src/types/index.ts` — TypeScript types
- `next.config.ts` — already configured with rewrite proxy
- `postcss.config.mjs` — already configured for Tailwind v4

### REMOVED
- `index.html` — Next.js generates HTML from `app/layout.tsx`
- `src/main.tsx` — replaced by `app/layout.tsx`
- `src/App.tsx` — replaced by `app/layout.tsx` + `app/page.tsx`
- `src/pages/UploadPage.tsx` — moved to `app/page.tsx`
- `src/pages/AnalyzePage.tsx` — moved to `app/analyze/page.tsx`
- `src/index.css` — renamed to `app/globals.css`
- `react-router-dom` dependency — replaced by Next.js file-based routing
- Vite-related devDependencies (`vitest` stays, `vite` removed)
- `dist/` build output — replaced by `.next/`

### MODIFIED
- `package.json` — scripts (dev/build/lint), remove vite/react-router deps
- `tsconfig.json` — add Next.js-specific options
- `src/test/test-utils.tsx` — remove react-router wrappers
- `src/test/setup.ts` — keep as-is (vitest-based, not affected by Next.js)

---

## Task 1: Update package.json — scripts and dependencies

**Files:**
- Modify: `src/frontend/package.json`

- [ ] **Step 1: Update scripts to use Next.js**

Replace the scripts section:

```json
"scripts": {
  "dev": "next dev --turbopack",
  "build": "next build",
  "start": "next start",
  "lint": "biome check .",
  "lint:fix": "biome check --write .",
  "test": "vitest",
  "typecheck": "tsc --noEmit"
}
```

- [ ] **Step 2: Remove Vite and React Router dependencies**

Remove from `dependencies`:
- `"react-router-dom": "^6.0.0"` (no longer needed — Next.js App Router)

Remove from `devDependencies`:
- None — vitest stays for testing

Note: `vite` was never explicitly listed (it was in the old package.json that was deleted). The `react-router-dom` import needs to be removed from source code.

- [ ] **Step 3: Install dependencies**

Run: `cd src/frontend && bun install`

- [ ] **Step 4: Verify no react-router-dom references remain**

Run: `grep -r "react-router" src/frontend/src/`

Expected: No output (all references will be removed in Tasks 3-4)

- [ ] **Step 5: Commit**

```bash
git add src/frontend/package.json src/frontend/bun.lock
git commit -m "chore(frontend): update scripts and deps for Next.js migration"
```

---

## Task 2: Update tsconfig.json for Next.js

**Files:**
- Modify: `src/frontend/tsconfig.json`

- [ ] **Step 1: Update tsconfig.json**

Replace the entire file with Next.js-compatible configuration:

```json
{
  "compilerOptions": {
    "target": "ES2017",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [{ "name": "next" }],
    "paths": { "@/*": ["./src/*"] }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules", "src/test"]
}
```

Changes from current:
- `"jsx": "react-jsx"` → `"jsx": "preserve"` (Next.js requires preserve)
- `"incremental": true` stays (already present)

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd src/frontend && npx tsc --noEmit`

Expected: Errors about missing app/ directory files — that's expected, we create them in Task 3.

- [ ] **Step 3: Commit**

```bash
git add src/frontend/tsconfig.json
git commit -m "chore(frontend): update tsconfig for Next.js (jsx: preserve)"
```

---

## Task 3: Create app/layout.tsx — root layout with providers

**Files:**
- Create: `src/frontend/src/app/layout.tsx`
- Create: `src/frontend/src/app/globals.css` (copy from index.css)

- [ ] **Step 1: Create globals.css by copying index.css**

Run: `cp src/frontend/src/index.css src/frontend/src/app/globals.css`

No changes needed to the CSS content — Tailwind v4 + shadcn theme stays identical.

- [ ] **Step 2: Create app/layout.tsx**

This replaces `main.tsx` + `App.tsx`. It sets up the HTML shell, QueryClientProvider, and shared header.

```tsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import type { Metadata } from "next"
import "./globals.css"

export const metadata: Metadata = {
  title: "AI Тренер — Фигурное катание",
  description: "ML-based AI coach for figure skating",
}

const queryClient = new QueryClient()

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ru" suppressHydrationWarning>
      <body className="min-h-screen bg-background text-foreground">
        <QueryClientProvider client={queryClient}>
          <header className="border-b border-border px-6 py-3">
            <h1 className="text-lg font-semibold">AI Тренер — Фигурное катание</h1>
          </header>
          <main>{children}</main>
        </QueryClientProvider>
      </body>
    </html>
  )
}
```

Key decisions:
- `QueryClient` created at module level (not inside component) to avoid re-creation on every render. This is the standard pattern for Next.js + React Query.
- `suppressHydrationWarning` on `<html>` for next-themes compatibility (dark mode).
- No `"use client"` on layout — it's a Server Component by default. React Query's `QueryClientProvider` is a client component, so Next.js handles the boundary automatically.
- Metadata exported for SEO.

- [ ] **Step 3: Verify layout renders**

Run: `cd src/frontend && npx next build 2>&1 | head -20`

Expected: Build succeeds or fails only due to missing page.tsx (created in Task 4).

- [ ] **Step 4: Commit**

```bash
git add src/frontend/src/app/layout.tsx src/frontend/src/app/globals.css
git commit -m "feat(frontend): add Next.js root layout with QueryClientProvider"
```

---

## Task 4: Create app/page.tsx — migrate UploadPage

**Files:**
- Create: `src/frontend/src/app/page.tsx`
- Delete: `src/frontend/src/pages/UploadPage.tsx` (after migration)

- [ ] **Step 1: Create app/page.tsx**

Copy the content of `src/frontend/src/pages/UploadPage.tsx` with these changes:

1. Add `"use client"` directive at the top (component uses useState, useCallback, useRef, useNavigate)
2. Replace `import { useNavigate } from "react-router-dom"` with `import { useRouter } from "next/navigation"`
3. Replace `const navigate = useNavigate()` with `const router = useRouter()`
4. Replace `navigate(\`/analyze?${params.toString()}\`)` with `router.push(\`/analyze?${params.toString()}\`)`
5. Remove `export default` → keep `export default function UploadPage()` (same)

```tsx
"use client"

import { AlertCircle, CheckCircle, Loader2, Upload } from "lucide-react"
import { useCallback, useRef, useState } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { detectPersons } from "@/lib/api"
import type { DetectResponse, PersonClick } from "@/types"

type Status = "idle" | "uploading" | "detecting" | "ready" | "error"

export default function UploadPage() {
  const router = useRouter()
  const fileRef = useRef<HTMLInputElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)

  const [file, setFile] = useState<File | null>(null)
  const [status, setStatus] = useState<Status>("idle")
  const [error, setError] = useState("")
  const [detectResult, setDetectResult] = useState<DetectResponse | null>(null)
  const [selectedPerson, setSelectedPerson] = useState<number | null>(null)
  const [clickCoord, setClickCoord] = useState<PersonClick | null>(null)

  // Settings
  const [frameSkip, setFrameSkip] = useState(1)
  const [layer, setLayer] = useState(3)
  const [tracking, setTracking] = useState("auto")
  const [doExport, setDoExport] = useState(true)

  // Derived: selected person's bbox as CSS percentage
  const selectedPersonData =
    detectResult && selectedPerson !== null
      ? detectResult.persons.find(p => p.track_id === selectedPerson)
      : undefined

  const handleFile = useCallback(
    async (f: File) => {
      setFile(f)
      setStatus("uploading")
      setError("")
      setDetectResult(null)
      setSelectedPerson(null)
      setClickCoord(null)

      const { data, error: err } = await detectPersons(f, tracking)
      if (err) {
        setError(err)
        setStatus("error")
        return
      }

      const resp = data as DetectResponse
      setDetectResult(resp)

      if (resp.auto_click) {
        setClickCoord(resp.auto_click)
        setSelectedPerson(resp.persons[0]?.track_id ?? 0)
      }

      setStatus("ready")
    },
    [tracking],
  )

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      const f = e.dataTransfer.files[0]
      if (f) handleFile(f)
    },
    [handleFile],
  )

  const pickPersonAt = useCallback(
    (clientX: number, clientY: number) => {
      if (!detectResult || !imgRef.current) return

      const rect = imgRef.current.getBoundingClientRect()
      const scaleX = imgRef.current.naturalWidth / rect.width
      const scaleY = imgRef.current.naturalHeight / rect.height
      const px = Math.round((clientX - rect.left) * scaleX)
      const py = Math.round((clientY - rect.top) * scaleY)

      let bestDist = Infinity
      let bestId = 0
      const natW = imgRef.current.naturalWidth
      const natH = imgRef.current.naturalHeight
      for (const p of detectResult.persons) {
        const hx = p.mid_hip[0] * natW
        const hy = p.mid_hip[1] * natH
        const dist = Math.hypot(px - hx, py - hy)
        if (dist < bestDist) {
          bestDist = dist
          bestId = p.track_id
        }
      }

      setClickCoord({ x: px, y: py })
      setSelectedPerson(bestId)
    },
    [detectResult],
  )

  const pickPerson = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      pickPersonAt(e.clientX, e.clientY)
    },
    [pickPersonAt],
  )

  const selectFromList = useCallback(
    (trackId: number) => {
      setSelectedPerson(trackId)
      const person = detectResult?.persons.find(p => p.track_id === trackId)
      if (person && imgRef.current) {
        const px = Math.round(person.mid_hip[0] * imgRef.current.naturalWidth)
        const py = Math.round(person.mid_hip[1] * imgRef.current.naturalHeight)
        setClickCoord({ x: px, y: py })
      }
    },
    [detectResult],
  )

  const handleAnalyze = () => {
    if (!detectResult || !clickCoord) return
    const params = new URLSearchParams({
      video_path: detectResult.video_path,
      person_click: `${clickCoord.x},${clickCoord.y}`,
      frame_skip: String(frameSkip),
      layer: String(layer),
      tracking,
      export: String(doExport),
    })
    router.push(`/analyze?${params.toString()}`)
  }

  const isAnalyzing = status === "uploading" || status === "detecting"
  const previewSrc = detectResult ? `data:image/png;base64,${detectResult.preview_image}` : ""

  // Compute bbox overlay CSS
  const bboxStyle = selectedPersonData
    ? {
        left: `${selectedPersonData.bbox[0] * 100}%`,
        top: `${selectedPersonData.bbox[1] * 100}%`,
        width: `${(selectedPersonData.bbox[2] - selectedPersonData.bbox[0]) * 100}%`,
        height: `${(selectedPersonData.bbox[3] - selectedPersonData.bbox[1]) * 100}%`,
      }
    : undefined

  return (
    <div className="mx-auto max-w-5xl p-6">
      {/* Upload zone */}
      {status === "idle" && (
        /* biome-ignore lint/a11y/useSemanticElements: div needed for drag-and-drop */
        <div
          role="button"
          tabIndex={0}
          onDragOver={e => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => fileRef.current?.click()}
          onKeyDown={e => {
            if (e.key === "Enter" || e.key === " ") fileRef.current?.click()
          }}
          className="flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-border p-12 transition-colors hover:border-primary"
        >
          <Upload className="mb-4 h-10 w-10 text-muted-foreground" />
          <p className="text-lg font-medium">Перетащите видео сюда или нажмите для выбора</p>
          <p className="mt-1 text-sm text-muted-foreground">MP4, MOV, WebM</p>
          <input
            ref={fileRef}
            type="file"
            accept="video/*"
            className="hidden"
            onChange={e => {
              const f = e.target.files?.[0]
              if (f) handleFile(f)
            }}
          />
        </div>
      )}

      {/* Loading */}
      {isAnalyzing && (
        <div className="flex flex-col items-center justify-center gap-4 py-20">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <p className="text-muted-foreground">Обнаружение людей...</p>
        </div>
      )}

      {/* Error */}
      {status === "error" && (
        <div className="flex flex-col items-center gap-4 py-10">
          <AlertCircle className="h-8 w-8 text-destructive" />
          <p className="text-destructive">{error}</p>
          <Button variant="outline" onClick={() => setStatus("idle")}>
            Попробовать снова
          </Button>
        </div>
      )}

      {/* Ready — two-panel layout */}
      {status === "ready" && detectResult && (
        <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
          {/* Left: preview */}
          <Card>
            <CardContent className="p-4">
              <div className="mb-3 flex items-center gap-2">
                {selectedPerson !== null && <CheckCircle className="h-4 w-4 text-green-500" />}
                <span className="text-sm">{detectResult.status}</span>
              </div>
              {/* biome-ignore lint/a11y/useSemanticElements: div needed as positioned container */}
              <div
                role="button"
                tabIndex={0}
                onClick={pickPerson}
                onKeyDown={e => {
                  if (e.key === "Enter" || e.key === " ") {
                    const rect = imgRef.current?.getBoundingClientRect()
                    if (rect) pickPersonAt(rect.left + rect.width / 2, rect.top + rect.height / 2)
                  }
                }}
                className="relative w-full cursor-crosshair"
              >
                <img
                  ref={imgRef}
                  src={previewSrc}
                  alt="Превью"
                  className="w-full rounded border border-border"
                  draggable={false}
                />
                {bboxStyle && (
                  <div
                    className="pointer-events-none absolute border-2 border-green-500 rounded-sm"
                    style={bboxStyle}
                  />
                )}
              </div>
              <p className="mt-2 text-xs text-muted-foreground">
                Нажмите на фигуриста на превью для выбора
              </p>
            </CardContent>
          </Card>

          {/* Right: settings */}
          <div className="flex flex-col gap-4">
            {/* Person list */}
            {detectResult.persons.length > 1 && (
              <Card>
                <CardContent className="p-4">
                  <h3 className="mb-2 text-sm font-medium">Фигуристы</h3>
                  <div className="flex flex-col gap-1">
                    {detectResult.persons.map(p => (
                      <button
                        type="button"
                        key={p.track_id}
                        onClick={() => selectFromList(p.track_id)}
                        className={`rounded px-3 py-1.5 text-left text-sm transition-colors ${
                          selectedPerson === p.track_id
                            ? "bg-primary text-primary-foreground"
                            : "hover:bg-muted"
                        }`}
                      >
                        #{p.track_id} — {p.hits} кадров
                      </button>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Settings */}
            <Card>
              <CardContent className="space-y-4 p-4">
                <h3 className="text-sm font-medium">Настройки</h3>

                <div>
                  <span className="mb-1 block text-xs text-muted-foreground">
                    Frame skip: {frameSkip}
                  </span>
                  <Slider
                    value={[frameSkip]}
                    onValueChange={([v]) => setFrameSkip(v)}
                    min={1}
                    max={8}
                    step={1}
                  />
                </div>

                <div>
                  <span className="mb-1 block text-xs text-muted-foreground">
                    HUD Layer: {layer}
                  </span>
                  <Slider
                    value={[layer]}
                    onValueChange={([v]) => setLayer(v)}
                    min={0}
                    max={3}
                    step={1}
                  />
                </div>

                <div>
                  <span className="mb-1 block text-xs text-muted-foreground">Трекинг</span>
                  <Select value={tracking} onValueChange={setTracking}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="auto">Авто</SelectItem>
                      <SelectItem value="none">Без трекинга</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex items-center gap-2">
                  <Checkbox
                    id="export"
                    checked={doExport}
                    onCheckedChange={v => setDoExport(v === true)}
                  />
                  <label htmlFor="export" className="text-sm">
                    Экспорт поз + CSV
                  </label>
                </div>
              </CardContent>
            </Card>

            {/* Actions */}
            <div className="flex flex-col gap-2">
              <Button onClick={handleAnalyze} disabled={clickCoord === null} size="lg">
                Анализировать
              </Button>
              <Button
                variant="outline"
                onClick={() => {
                  setStatus("idle")
                  setDetectResult(null)
                  setClickCoord(null)
                  setSelectedPerson(null)
                }}
              >
                Другое видео
              </Button>
            </div>

            {/* File info */}
            {file && <p className="text-xs text-muted-foreground">{file.name}</p>}
          </div>
        </div>
      )}
    </div>
  )
}
```

Diff summary vs original `UploadPage.tsx`:
- Line 1: Added `"use client"`
- Line 4: `react-router-dom` → `next/navigation`
- Line 5: `useNavigate` → `useRouter`
- Line 22: `useNavigate()` → `useRouter()`
- Line 133: `navigate(...)` → `router.push(...)`

- [ ] **Step 2: Verify build**

Run: `cd src/frontend && npx next build 2>&1 | tail -20`

Expected: Build succeeds for `/` route. `/analyze` route will fail — that's expected (Task 5).

- [ ] **Step 3: Commit**

```bash
git add src/frontend/src/app/page.tsx
git commit -m "feat(frontend): migrate UploadPage to Next.js App Router"
```

---

## Task 5: Create app/analyze/page.tsx — migrate AnalyzePage

**Files:**
- Create: `src/frontend/src/app/analyze/page.tsx`
- Delete: `src/frontend/src/pages/AnalyzePage.tsx` (after migration)

- [ ] **Step 1: Create app/analyze/page.tsx**

Copy the content of `src/frontend/src/pages/AnalyzePage.tsx` with these changes:

1. Add `"use client"` directive at the top
2. Replace `import { useNavigate, useSearchParams } from "react-router-dom"` with `import { useRouter, useSearchParams } from "next/navigation"`
3. Replace `const navigate = useNavigate()` with `const router = useRouter()`
4. Replace all `navigate("/")` with `router.push("/")`
5. `useSearchParams()` stays the same — Next.js has the same hook

```tsx
"use client"

import { AlertCircle, ArrowLeft, CheckCircle, Download, Loader2 } from "lucide-react"
import { useCallback, useEffect, useMemo, useState } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { processVideo } from "@/lib/api"
import type { PersonClick, ProcessResponse } from "@/types"

type Phase = "processing" | "done" | "error"

export default function AnalyzePage() {
  const params = useSearchParams()
  const router = useRouter()

  const [phase, setPhase] = useState<Phase>("processing")
  const [progress, setProgress] = useState(0)
  const [message, setMessage] = useState("Начинаем...")
  const [result, setResult] = useState<ProcessResponse | null>(null)
  const [error, setError] = useState("")

  const videoPath = params.get("video_path") || ""
  const clickParts = (params.get("person_click") || "0,0").split(",")
  const personClick: PersonClick = useMemo(
    () => ({
      x: Number(clickParts[0]),
      y: Number(clickParts[1]),
    }),
    [clickParts],
  )
  const frameSkip = Number(params.get("frame_skip") || 1)
  const layer = Number(params.get("layer") || 3)
  const tracking = params.get("tracking") || "auto"
  const doExport = params.get("export") !== "false"

  const startProcessing = useCallback(() => {
    setPhase("processing")
    setProgress(0)
    setMessage("Подготовка...")

    processVideo(
      {
        video_path: videoPath,
        person_click: personClick,
        frame_skip: frameSkip,
        layer: layer,
        tracking: tracking,
        export: doExport,
      },
      {
        onProgress(p, msg) {
          setProgress(Math.round(p * 100))
          setMessage(msg)
        },
        onResult(r) {
          setResult(r as ProcessResponse)
          setPhase("done")
        },
        onError(err) {
          setError(err)
          setPhase("error")
        },
      },
    )
  }, [videoPath, personClick, frameSkip, layer, tracking, doExport])

  useEffect(() => {
    if (videoPath) startProcessing()
  }, [videoPath, startProcessing])

  const videoUrl = result ? `/api/outputs/${result.video_path}` : ""
  const posesUrl = result?.poses_path ? `/api/outputs/${result.poses_path}` : null
  const csvUrl = result?.csv_path ? `/api/outputs/${result.csv_path}` : null

  return (
    <div className="mx-auto max-w-4xl p-6">
      <Button variant="ghost" onClick={() => router.push("/")} className="mb-4 gap-1">
        <ArrowLeft className="h-4 w-4" />
        Назад
      </Button>

      {/* Processing */}
      {phase === "processing" && (
        <Card>
          <CardContent className="flex flex-col items-center gap-4 p-8">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
            <h2 className="text-lg font-medium">Анализ видео</h2>
            <Progress value={progress} className="w-full max-w-md" />
            <p className="text-sm text-muted-foreground">{message}</p>
            <p className="text-xs text-muted-foreground">{progress}%</p>
          </CardContent>
        </Card>
      )}

      {/* Done */}
      {phase === "done" && result && (
        <div className="space-y-4">
          <Card>
            <CardContent className="p-4">
              <div className="mb-2 flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-green-500" />
                <h2 className="font-medium">{result.status}</h2>
              </div>
              <div className="grid grid-cols-2 gap-2 text-sm text-muted-foreground sm:grid-cols-4">
                <span>Кадров: {result.stats.total_frames}</span>
                <span>Валидных: {result.stats.valid_frames}</span>
                <span>FPS: {result.stats.fps}</span>
                <span>{result.stats.resolution}</span>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              {/* biome-ignore lint/a11y/useMediaCaption: analysis output, not media */}
              <video src={videoUrl} controls className="w-full rounded border border-border" />
            </CardContent>
          </Card>

          <div className="flex flex-wrap gap-2">
            <Button variant="outline" size="sm" asChild>
              <a href={videoUrl} download>
                <Download className="mr-1 h-4 w-4" />
                Видео
              </a>
            </Button>
            {posesUrl && (
              <Button variant="outline" size="sm" asChild>
                <a href={posesUrl} download>
                  <Download className="mr-1 h-4 w-4" />
                  Позы (.npy)
                </a>
              </Button>
            )}
            {csvUrl && (
              <Button variant="outline" size="sm" asChild>
                <a href={csvUrl} download>
                  <Download className="mr-1 h-4 w-4" />
                  Биомеханика (.csv)
                </a>
              </Button>
            )}
          </div>
        </div>
      )}

      {/* Error */}
      {phase === "error" && (
        <Card>
          <CardContent className="flex flex-col items-center gap-4 p-8">
            <AlertCircle className="h-8 w-8 text-destructive" />
            <p className="text-destructive">{error}</p>
            <div className="flex gap-2">
              <Button onClick={startProcessing}>Повторить</Button>
              <Button variant="outline" onClick={() => router.push("/")}>
                Назад
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
```

Diff summary vs original `AnalyzePage.tsx`:
- Line 1: Added `"use client"`
- Line 4: `react-router-dom` → `next/navigation`
- Line 4: `useNavigate` → `useRouter`
- Line 13: `const navigate = useNavigate()` → `const router = useRouter()`
- Line 13: `const [params] = useSearchParams()` → `const params = useSearchParams()` (Next.js returns read-only URLSearchParams, no destructuring needed)
- Lines 77, 155: `navigate("/")` → `router.push("/")`

- [ ] **Step 2: Verify build**

Run: `cd src/frontend && npx next build 2>&1 | tail -20`

Expected: Build succeeds for both `/` and `/analyze` routes.

- [ ] **Step 3: Commit**

```bash
git add src/frontend/src/app/analyze/page.tsx
git commit -m "feat(frontend): migrate AnalyzePage to Next.js App Router"
```

---

## Task 6: Clean up — remove Vite artifacts and old routing code

**Files:**
- Delete: `src/frontend/index.html`
- Delete: `src/frontend/src/main.tsx`
- Delete: `src/frontend/src/App.tsx`
- Delete: `src/frontend/src/pages/UploadPage.tsx`
- Delete: `src/frontend/src/pages/AnalyzePage.tsx`
- Delete: `src/frontend/src/index.css` (moved to `app/globals.css`)
- Delete: `src/frontend/src/pages/` directory

- [ ] **Step 1: Delete old Vite entry files**

```bash
rm src/frontend/index.html
rm src/frontend/src/main.tsx
rm src/frontend/src/App.tsx
rm src/frontend/src/index.css
rm -rf src/frontend/src/pages/
```

- [ ] **Step 2: Verify no react-router-dom references remain in src/**

Run: `grep -r "react-router" src/frontend/src/`

Expected: No output

- [ ] **Step 3: Verify no references to deleted files remain**

Run: `grep -r "main.tsx\|App.tsx\|pages/Upload\|pages/Analyze\|index.css" src/frontend/src/`

Expected: No output (all imports now point to `@/components/ui/*`, `@/lib/*`, `@/types/*`)

- [ ] **Step 4: Verify build**

Run: `cd src/frontend && npx next build 2>&1 | tail -20`

Expected: Build succeeds with no errors.

- [ ] **Step 5: Commit**

```bash
git add -A src/frontend/
git commit -m "refactor(frontend): remove Vite entry files and react-router-dom"
```

---

## Task 7: Update test infrastructure

**Files:**
- Modify: `src/frontend/src/test/test-utils.tsx`

- [ ] **Step 1: Update test-utils.tsx**

The current `renderWithProviders` doesn't wrap with any router (no react-router). With Next.js, we need to mock `next/navigation` hooks. Since tests use MSW and vitest (not Next.js test runner), the components are imported directly.

Update `test-utils.tsx`:

```tsx
import { type RenderOptions, render } from "@testing-library/react"
import type { ReactElement } from "react"

// Mock next/navigation for tests
vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
    prefetch: vi.fn(),
  }),
  useSearchParams: () => new URLSearchParams(),
  usePathname: () => "/",
}))

// Custom render with providers (add QueryClientProvider later if needed)
export function renderWithProviders(ui: ReactElement, options?: Omit<RenderOptions, "wrapper">) {
  return render(ui, options)
}

// Re-export everything from RTL
export * from "@testing-library/react"
```

Note: If `vi.mock` is not available at module scope in vitest, move it into a separate `__mocks__/next-navigation.ts` file or use `vi.hoisted()`. The exact approach depends on vitest configuration.

Alternative (simpler, no vi.mock at module scope):

```tsx
// src/test/__mocks__/next-navigation.ts
import { vi } from "vitest"

export const useRouter = vi.fn(() => ({
  push: vi.fn(),
  replace: vi.fn(),
  back: vi.fn(),
  forward: vi.fn(),
  refresh: vi.fn(),
  prefetch: vi.fn(),
}))

export const useSearchParams = vi.fn(() => new URLSearchParams())

export const usePathname = vi.fn(() => "/")
```

Then in `vitest.config` or `setup.ts`, add:
```ts
vi.mock("next/navigation", () => import("./__mocks__/next-navigation"))
```

- [ ] **Step 2: Verify tests still pass**

Run: `cd src/frontend && npx vitest run`

Expected: All existing tests pass (there may be few or no component tests currently).

- [ ] **Step 3: Commit**

```bash
git add src/frontend/src/test/
git commit -m "test(frontend): mock next/navigation for vitest"
```

---

## Task 8: Final verification — dev server + build + lint

**Files:** None (verification only)

- [ ] **Step 1: Run dev server and verify both routes work**

Run: `cd src/frontend && npx next dev --turbopack`

Manual checks:
1. Open `http://localhost:3000` — should show upload page
2. Upload a test video — should detect persons
3. Click "Анализировать" — should navigate to `/analyze?...`
4. Analyze page should show progress and results
5. Back button should return to `/`

- [ ] **Step 2: Run production build**

Run: `cd src/frontend && npx next build`

Expected: Build succeeds, shows route table:
```
Route (app)            Size    First Load JS
┌ ○ /                  X kB    Y kB
└ ○ /analyze           X kB    Y kB
```

- [ ] **Step 3: Run lint**

Run: `cd src/frontend && npx biome check .`

Expected: No new errors (only pre-existing ones).

- [ ] **Step 4: Run tests**

Run: `cd src/frontend && npx vitest run`

Expected: All tests pass.

- [ ] **Step 5: Final commit (if any fixes were needed)**

```bash
git add -A src/frontend/
git commit -m "fix(frontend): address issues from final verification"
```

---

## Self-Review Checklist

### Spec Coverage
- [x] Vite → Next.js migration: Tasks 1-6
- [x] React Router → App Router: Tasks 4-5
- [x] API proxy maintained: next.config.ts rewrite already exists
- [x] Tailwind v4 + shadcn preserved: no CSS changes needed
- [x] Tests updated: Task 7
- [x] Build verification: Task 8

### Placeholder Scan
- No TBDs, TODOs, or "implement later" found
- All code blocks contain complete, copy-pasteable code
- All file paths are exact

### Type Consistency
- `useRouter` from `next/navigation` used consistently in Tasks 4 and 5
- `useSearchParams` from `next/navigation` used consistently
- `router.push()` used consistently (not `navigate()`)
- `QueryClient` instantiation at module level in layout.tsx

### Potential Issues
1. **`useSearchParams` in Next.js** — requires wrapping in `Suspense` boundary when used in a page that's not explicitly a client component. Since both pages have `"use client"`, this is fine. But if Next.js warns, wrap in `<Suspense>`.
2. **SSE streaming** — the `processVideo()` function uses `res.body.getReader()` which works in the browser. No change needed.
3. **`output: "standalone"` in next.config.ts** — good for deployment, works with `next start`.
4. **Image optimization** — the preview image uses a base64 data URI, not `next/image`. This is intentional for dynamically loaded images. No change needed.
5. **`react-router-dom` removal** — must be removed from `package.json` dependencies. Task 1 covers this.
