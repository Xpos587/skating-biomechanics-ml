# Responsive UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all pages fully responsive across iPhone (375px), iPad (768px), and MacBook (1440px) with proper layouts, touch targets, and content scaling.

**Architecture:** Systematic pass through every page and component, applying Tailwind responsive breakpoints (`sm:640px`, `md:768px`, `lg:1024px`). Content containers scale from narrow on mobile to wide on desktop. Grids collapse gracefully. Touch targets meet 44px minimum on mobile.

**Tech Stack:** Next.js 16, Tailwind CSS, lucide-react

---

## Breakpoint Strategy

| Device | Width | Tailwind | Content max-w | Grid |
|--------|-------|----------|---------------|------|
| iPhone SE | 375px | default | full | 2-3 cols |
| iPhone 14 | 390px | default | full | 2-3 cols |
| iPad Mini | 768px | `md:` | `max-w-2xl` | 3 cols |
| iPad Pro 12.9 | 1024px | `lg:` | `max-w-4xl` | 4 cols |
| MacBook | 1440px | `lg:` | `max-w-5xl` | 4 cols |

---

## File Structure

### Modify (pages)
| File | Change |
|------|--------|
| `src/frontend/src/app/(app)/layout.tsx` | Responsive main container padding |
| `src/frontend/src/app/(app)/feed/page.tsx` | Responsive max-width |
| `src/frontend/src/app/(app)/upload/page.tsx` | Responsive max-width, mode buttons |
| `src/frontend/src/app/(app)/progress/page.tsx` | Responsive element grid (8 items) |
| `src/frontend/src/app/(app)/dashboard/page.tsx` | Responsive max-width |
| `src/frontend/src/app/(app)/sessions/[id]/page.tsx` | Responsive max-width, video |
| `src/frontend/src/app/(app)/students/[id]/page.tsx` | Responsive element grid, tab bar |
| `src/frontend/src/app/(app)/connections/page.tsx` | Responsive max-width |
| `src/frontend/src/app/(app)/profile/page.tsx` | Responsive form grid |
| `src/frontend/src/app/(auth)/layout.tsx` | Safe area padding for notch phones |

### Modify (components)
| File | Change |
|------|--------|
| `src/frontend/src/components/layout/bottom-dock.tsx` | Safe area padding, larger touch targets |
| `src/frontend/src/components/upload/element-picker.tsx` | Responsive grid cols |
| `src/frontend/src/components/progress/trend-chart.tsx` | Responsive chart height |
| `src/frontend/src/components/session/session-card.tsx` | Responsive padding |

---

## Task 1: App layout — responsive container

**Files:**
- Modify: `src/frontend/src/app/(app)/layout.tsx`

- [ ] **Step 1: Update (app)/layout.tsx**

Replace the `<main>` element to use responsive padding and remove fixed bottom padding (handled by individual pages + bottom dock safe area):

```tsx
import { Activity } from "lucide-react"
import { cookies } from "next/headers"
import Link from "next/link"
import { redirect } from "next/navigation"
import { getTranslations } from "next-intl/server"
import { AppNav } from "@/components/app-nav"
import { BottomDock } from "@/components/layout/bottom-dock"

export default async function AppLayout({ children }: { children: React.ReactNode }) {
  const t = await getTranslations("app")

  // Server-side auth gate — redirect to login if no auth cookie
  const hasAuth = (await cookies()).get("sb_auth")?.value
  if (!hasAuth) redirect("/login")

  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-50 border-b border-border bg-background">
        <div className="mx-auto flex h-[52px] max-w-5xl items-center justify-between px-4 sm:px-6">
          <Link href="/feed" className="flex items-center gap-2 font-semibold">
            <Activity className="h-5 w-5" />
            <span className="hidden sm:inline">{t("title")}</span>
          </Link>
          <AppNav />
        </div>
      </header>
      <main className="mx-auto w-full max-w-5xl flex-1 px-4 py-4 pb-24 sm:px-6 sm:py-6 sm:pb-8">{children}</main>
      <BottomDock />
    </div>
  )
}
```

Changes: header content wrapped in `max-w-5xl mx-auto`, main uses `max-w-5xl` (1024px — good for both iPad and MacBook), responsive padding (`py-4` mobile → `sm:py-6` desktop).

- [ ] **Step 2: Commit**

```bash
git add src/frontend/src/app/\(app\)/layout.tsx
git commit -m "style(frontend): responsive app layout container"
```

---

## Task 2: Bottom dock — safe area + touch targets

**Files:**
- Modify: `src/frontend/src/components/layout/bottom-dock.tsx`

- [ ] **Step 1: Update bottom-dock.tsx**

```tsx
"use client"

import { BarChart3, Camera, Newspaper, Users } from "lucide-react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { useQuery } from "@tanstack/react-query"
import { z } from "zod"
import { apiFetch } from "@/lib/api-client"

const RelationshipListSchema = z.object({
  relationships: z.array(z.object({ status: z.string() })),
})

export function BottomDock() {
  const pathname = usePathname()

  const { data: relsData } = useQuery({
    queryKey: ["relationships"],
    queryFn: () => apiFetch("/relationships", RelationshipListSchema),
  })
  const hasStudents = (relsData?.relationships ?? []).some((r) => r.status === "active")

  const tabs = [
    { href: "/feed", icon: Newspaper, label: "Лента" },
    { href: "/upload", icon: Camera, label: "Запись" },
    { href: "/progress", icon: BarChart3, label: "Прогресс" },
    ...(hasStudents
      ? [{ href: "/dashboard", icon: Users, label: "Ученики" }]
      : []),
  ] as const

  const isActive = (href: string) =>
    pathname === href || pathname.startsWith(href + "/")

  return (
    <nav className="fixed inset-x-0 bottom-0 z-50 border-t border-border bg-background pb-[env(safe-area-inset-bottom)] md:hidden">
      <div className="flex h-16 items-center justify-around px-2">
        {tabs.map((tab) => {
          const Icon = tab.icon
          const active = isActive(tab.href)
          return (
            <Link
              key={tab.href}
              href={tab.href}
              className={`flex flex-col items-center gap-0.5 rounded-lg px-4 py-1.5 text-[10px] transition-colors ${
                active ? "text-foreground" : "text-muted-foreground"
              }`}
            >
              <Icon className="h-5 w-5" />
              <span>{tab.label}</span>
            </Link>
          )
        })}
      </div>
    </nav>
  )
}
```

Changes:
- `pb-[env(safe-area-inset-bottom)]` — iPhone notch/home indicator padding
- `px-2` on container, `px-4 py-1.5` on each tab — larger 44px+ touch targets
- `rounded-lg` on active area for visual tap feedback

- [ ] **Step 2: Commit**

```bash
git add src/frontend/src/components/layout/bottom-dock.tsx
git commit -m "style(frontend): bottom dock safe area and touch targets"
```

---

## Task 3: Auth layout — safe area for notch

**Files:**
- Modify: `src/frontend/src/app/(auth)/layout.tsx`

- [ ] **Step 1: Update (auth)/layout.tsx**

```tsx
import { Activity } from "lucide-react"
import { cookies } from "next/headers"
import Link from "next/link"
import { redirect } from "next/navigation"
import { getTranslations } from "next-intl/server"

export default async function AuthLayout({ children }: { children: React.ReactNode }) {
  const t = await getTranslations("app")

  // Already authenticated — redirect to app
  const hasAuth = (await cookies()).get("sb_auth")?.value
  if (hasAuth) redirect("/feed")

  return (
    <div className="flex min-h-[dvh] flex-col">
      <header className="border-b border-border bg-background pt-[env(safe-area-inset-top)]">
        <div className="flex h-[52px] items-center px-4">
          <Link href="/" className="flex items-center gap-2 font-semibold">
            <Activity className="h-5 w-5" />
            <span>{t("title")}</span>
          </Link>
        </div>
      </header>
      <div className="flex flex-1 items-center justify-center px-4 py-8">
        <div className="w-full max-w-md">{children}</div>
      </div>
    </div>
  )
}
```

Changes: `min-h-[dvh]` for correct mobile viewport height (avoids address bar issues), `pt-[env(safe-area-inset-top)]` for notch, `py-8` vertical padding.

- [ ] **Step 2: Commit**

```bash
git add src/frontend/src/app/\(auth\)/layout.tsx
git commit -m "style(frontend): auth layout safe area and dvh"
```

---

## Task 4: Feed page — responsive width

**Files:**
- Modify: `src/frontend/src/app/(app)/feed/page.tsx`

- [ ] **Step 1: Update feed/page.tsx**

```tsx
"use client"

import { useSessions } from "@/lib/api/sessions"
import { SessionCard } from "@/components/session/session-card"

export default function FeedPage() {
  const { data, isLoading } = useSessions()

  if (isLoading) {
    return <div className="flex items-center justify-center py-20 text-muted-foreground">Загрузка...</div>
  }

  if (!data?.sessions.length) {
    return (
      <div className="text-center py-20">
        <p className="text-muted-foreground">Нет записей</p>
        <p className="text-sm text-muted-foreground mt-1">Загрузите первое видео</p>
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-2xl space-y-3 sm:max-w-3xl">
      {data.sessions.map((session) => (
        <SessionCard key={session.id} session={session} />
      ))}
    </div>
  )
}
```

Changes: `max-w-lg` → `max-w-2xl sm:max-w-3xl`. On mobile, full width. On tablet/desktop, comfortable reading width.

- [ ] **Step 2: Commit**

```bash
git add src/frontend/src/app/\(app\)/feed/page.tsx
git commit -m "style(frontend): responsive feed page width"
```

---

## Task 5: Upload page — responsive layout

**Files:**
- Modify: `src/frontend/src/app/(app)/upload/page.tsx`

- [ ] **Step 1: Update upload/page.tsx**

```tsx
"use client"

import { useRef, useState } from "react"
import { useRouter } from "next/navigation"
import { toast } from "sonner"
import { useCreateSession } from "@/lib/api/sessions"
import { CameraRecorder } from "@/components/upload/camera-recorder"
import { ChunkedUploader } from "@/components/upload/chunked-uploader"
import { ElementPicker } from "@/components/upload/element-picker"

type Mode = "pick" | "record" | "uploading"

export default function UploadPage() {
  const router = useRouter()
  const createSession = useCreateSession()
  const [mode, setMode] = useState<Mode>("pick")
  const [elementType, setElementType] = useState<string | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  const handleFile = (f: File) => {
    setFile(f)
    setMode("uploading")
  }

  const handleRecorded = (blob: Blob) => {
    const f = new File([blob], `recording_${Date.now()}.webm`, { type: blob.type })
    setFile(f)
    setMode("uploading")
  }

  const handleUploaded = async (key: string) => {
    if (!elementType) return
    try {
      await createSession.mutateAsync({ element_type: elementType })
      toast.success("Видео загружено, анализ начат")
      router.push("/feed")
    } catch {
      toast.error("Ошибка создания сессии")
    }
  }

  return (
    <div className="mx-auto max-w-lg space-y-6">
      {mode !== "uploading" && (
        <div className="flex gap-3">
          <button onClick={() => setMode("pick")} className={`flex-1 rounded-xl border p-3 sm:p-4 text-center text-sm ${mode === "pick" ? "border-primary bg-primary/10" : "border-border"}`}>
            Выбрать файл
          </button>
          <button onClick={() => setMode("record")} className={`flex-1 rounded-xl border p-3 sm:p-4 text-center text-sm ${mode === "record" ? "border-primary bg-primary/10" : "border-border"}`}>
            Записать
          </button>
        </div>
      )}

      {mode === "pick" && (
        <input ref={fileRef} type="file" accept="video/*" className="hidden" onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])} />
      )}

      {mode === "record" && <CameraRecorder onRecorded={handleRecorded} />}

      {mode !== "uploading" && (
        <div className="space-y-2">
          <p className="text-sm font-medium">Элемент:</p>
          <ElementPicker value={elementType} onChange={setElementType} />
        </div>
      )}

      {mode === "pick" && (
        <button onClick={() => fileRef.current?.click()} className="w-full rounded-xl bg-primary text-primary-foreground py-3 text-sm font-medium">
          Выбрать видео
        </button>
      )}

      {mode === "uploading" && file && <ChunkedUploader file={file} onUploaded={handleUploaded} />}
    </div>
  )
}
```

Changes: `p-4` → `p-3 sm:p-4` on mode toggle buttons for mobile.

- [ ] **Step 2: Commit**

```bash
git add src/frontend/src/app/\(app\)/upload/page.tsx
git commit -m "style(frontend): responsive upload page"
```

---

## Task 6: Progress page — responsive element grid

**Files:**
- Modify: `src/frontend/src/app/(app)/progress/page.tsx`

- [ ] **Step 1: Update progress/page.tsx**

```tsx
"use client"

import { useState } from "react"
import { useMetricRegistry, useTrend } from "@/lib/api/metrics"
import { TrendChart } from "@/components/progress/trend-chart"
import { PeriodSelector } from "@/components/progress/period-selector"

const ELEMENTS = [
  { id: "three_turn", label: "Тройка" }, { id: "waltz_jump", label: "Вальсовый" },
  { id: "toe_loop", label: "Перекидной" }, { id: "flip", label: "Флип" },
  { id: "salchow", label: "Сальхов" }, { id: "loop", label: "Петля" },
  { id: "lutz", label: "Лютц" }, { id: "axel", label: "Аксель" },
]

export default function ProgressPage() {
  const { data: registry } = useMetricRegistry()
  const [element, setElement] = useState("waltz_jump")
  const [metric, setMetric] = useState("max_height")
  const [period, setPeriod] = useState("30d")
  const { data: trend } = useTrend(undefined, element, metric, period)

  const availableMetrics = registry
    ? Object.entries(registry).filter(([, v]) => (v as any).element_types.includes(element))
    : []

  return (
    <div className="mx-auto max-w-2xl space-y-4 sm:max-w-3xl">
      <div className="grid grid-cols-4 gap-1.5 sm:gap-2">
        {ELEMENTS.map((el) => (
          <button
            key={el.id}
            onClick={() => setElement(el.id)}
            className={`truncate rounded-xl border p-1.5 text-center text-[11px] sm:p-2 sm:text-xs ${element === el.id ? "border-primary bg-primary/10" : "border-border"}`}
          >
            {el.label}
          </button>
        ))}
      </div>

      <div className="space-y-2">
        <select
          value={metric}
          onChange={(e) => setMetric(e.target.value)}
          className="w-full rounded-xl border border-border bg-background px-3 py-2.5 text-sm"
        >
          {availableMetrics.map(([name, def]) => (
            <option key={name} value={name}>{(def as any).label_ru}</option>
          ))}
        </select>
        <PeriodSelector value={period} onChange={setPeriod} />
      </div>

      {trend && <TrendChart data={trend} />}
    </div>
  )
}
```

Changes:
- `max-w-2xl sm:max-w-3xl` — scales from mobile to desktop
- `gap-1.5 sm:gap-2` — tighter gaps on mobile
- `p-1.5 text-[11px] sm:p-2 sm:text-xs` — smaller buttons on mobile, `truncate` prevents overflow
- `py-2.5` on select for better mobile touch target

- [ ] **Step 2: Commit**

```bash
git add src/frontend/src/app/\(app\)/progress/page.tsx
git commit -m "style(frontend): responsive progress page grid"
```

---

## Task 7: Dashboard page — responsive width

**Files:**
- Modify: `src/frontend/src/app/(app)/dashboard/page.tsx`

- [ ] **Step 1: Update dashboard/page.tsx**

```tsx
"use client"

import { useRelationships } from "@/lib/api/relationships"
import { StudentCard } from "@/components/coach/student-card"

export default function DashboardPage() {
  const { data, isLoading } = useRelationships()

  const students = (data?.relationships ?? []).filter(
    (r) => r.status === "active",
  )

  if (isLoading) return <div className="py-20 text-center text-muted-foreground">Загрузка...</div>

  if (!students.length) {
    return (
      <div className="text-center py-20">
        <p className="text-muted-foreground">Нет учеников</p>
        <p className="text-sm text-muted-foreground mt-1">Пригласите первого ученика</p>
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-2xl space-y-3 sm:max-w-3xl">
      <h1 className="text-lg font-semibold">Ученики</h1>
      {students.map((rel) => (
        <StudentCard key={rel.id} rel={rel} />
      ))}
    </div>
  )
}
```

Changes: `max-w-lg` → `max-w-2xl sm:max-w-3xl`.

- [ ] **Step 2: Commit**

```bash
git add src/frontend/src/app/\(app\)/dashboard/page.tsx
git commit -m "style(frontend): responsive dashboard page width"
```

---

## Task 8: Session detail page — responsive layout

**Files:**
- Modify: `src/frontend/src/app/(app)/sessions/[id]/page.tsx`

- [ ] **Step 1: Update sessions/[id]/page.tsx**

```tsx
"use client"

import { useParams } from "next/navigation"
import { useSession } from "@/lib/api/sessions"
import { MetricRow } from "@/components/session/metric-row"

const ELEMENT_NAMES: Record<string, string> = {
  three_turn: "Тройка", waltz_jump: "Вальсовый", toe_loop: "Перекидной",
  flip: "Флип", salchow: "Сальхов", loop: "Петля",
  lutz: "Лютц", axel: "Аксель",
}

export default function SessionDetailPage() {
  const { id } = useParams<{ id: string }>()
  const { data: session, isLoading } = useSession(id)

  if (isLoading) return <div className="py-20 text-center text-muted-foreground">Загрузка...</div>
  if (!session) return <div className="py-20 text-center text-muted-foreground">Сессия не найдена</div>

  return (
    <div className="mx-auto max-w-2xl space-y-6 sm:max-w-3xl">
      <div>
        <h1 className="text-xl font-semibold">{ELEMENT_NAMES[session.element_type] ?? session.element_type}</h1>
        <p className="text-sm text-muted-foreground">{new Date(session.created_at).toLocaleDateString("ru-RU")}</p>
      </div>

      {session.processed_video_url && (
        <video src={session.processed_video_url} controls className="w-full rounded-xl" />
      )}

      {session.metrics.length > 0 && (
        <div className="rounded-2xl border border-border p-3 sm:p-4">
          <h2 className="text-sm font-medium mb-2">Метрики</h2>
          {session.metrics.map((m) => (
            <MetricRow
              key={m.id}
              name={m.metric_name}
              label={m.metric_name}
              value={m.metric_value}
              unit={m.unit ?? (m.metric_name === "score" ? "" : m.metric_name === "deg" ? "°" : "")}
              isInRange={m.is_in_range}
              isPr={m.is_pr}
              prevBest={m.prev_best}
              refRange={m.reference_value ? [m.reference_value, m.reference_value + 1] : null}
            />
          ))}
        </div>
      )}

      {session.recommendations && session.recommendations.length > 0 && (
        <div className="rounded-2xl border border-border p-3 sm:p-4">
          <h2 className="text-sm font-medium mb-2">Рекомендации</h2>
          <ul className="space-y-1 text-sm text-muted-foreground">
            {session.recommendations.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
```

Changes: `max-w-lg` → `max-w-2xl sm:max-w-3xl`, `p-4` → `p-3 sm:p-4` on metric/recommendation cards.

- [ ] **Step 2: Commit**

```bash
git add src/frontend/src/app/\(app\)/sessions/\[id\]/page.tsx
git commit -m "style(frontend): responsive session detail page"
```

---

## Task 9: Student profile page — responsive grid + tabs

**Files:**
- Modify: `src/frontend/src/app/(app)/students/[id]/page.tsx`

- [ ] **Step 1: Update students/[id]/page.tsx**

```tsx
"use client"

import { useState } from "react"
import { useParams } from "next/navigation"
import Link from "next/link"
import { useDiagnostics, useTrend } from "@/lib/api/metrics"
import { DiagnosticsList } from "@/components/coach/diagnostics-list"
import { TrendChart } from "@/components/progress/trend-chart"
import { PeriodSelector } from "@/components/progress/period-selector"

const ELEMENTS = [
  { id: "three_turn", label: "Тройка" }, { id: "waltz_jump", label: "Вальсовый" },
  { id: "toe_loop", label: "Перекидной" }, { id: "flip", label: "Флип" },
  { id: "salchow", label: "Сальхов" }, { id: "loop", label: "Петля" },
  { id: "lutz", label: "Лютц" }, { id: "axel", label: "Аксель" },
]

export default function StudentProfilePage() {
  const { id } = useParams<{ id: string }>()
  const [tab, setTab] = useState<"progress" | "diagnostics">("progress")
  const [element, setElement] = useState("waltz_jump")
  const [metric, setMetric] = useState("max_height")
  const [period, setPeriod] = useState("30d")

  const { data: trend } = useTrend(id, element, metric, period)
  const { data: diag } = useDiagnostics(id)

  return (
    <div className="mx-auto max-w-2xl space-y-4 sm:max-w-3xl">
      <div className="flex gap-2">
        <Link href="/dashboard" className="text-sm text-muted-foreground hover:underline">&larr; Назад</Link>
      </div>

      <div className="flex gap-1 rounded-lg bg-muted p-1">
        <button onClick={() => setTab("progress")} className={`flex-1 rounded-md px-3 py-2 text-sm font-medium ${tab === "progress" ? "bg-background shadow-sm" : ""}`}>
          Прогресс
        </button>
        <button onClick={() => setTab("diagnostics")} className={`flex-1 rounded-md px-3 py-2 text-sm font-medium ${tab === "diagnostics" ? "bg-background shadow-sm" : ""}`}>
          Диагностика
        </button>
      </div>

      {tab === "progress" && (
        <div className="space-y-4">
          <div className="grid grid-cols-4 gap-1.5 sm:gap-2">
            {ELEMENTS.map((el) => (
              <button key={el.id} onClick={() => setElement(el.id)} className={`truncate rounded-xl border p-1.5 text-center text-[11px] sm:p-2 sm:text-xs ${element === el.id ? "border-primary bg-primary/10" : "border-border"}`}>
                {el.label}
              </button>
            ))}
          </div>
          <select value={metric} onChange={(e) => setMetric(e.target.value)} className="w-full rounded-xl border border-border bg-background px-3 py-2.5 text-sm">
            <option value="max_height">Высота прыжка</option>
            <option value="airtime">Время полёта</option>
            <option value="landing_knee_stability">Стабильность приземления</option>
            <option value="rotation_speed">Скорость вращения</option>
          </select>
          <PeriodSelector value={period} onChange={setPeriod} />
          {trend && <TrendChart data={trend} />}
        </div>
      )}

      {tab === "diagnostics" && diag && <DiagnosticsList findings={diag.findings} />}
    </div>
  )
}
```

Changes: `max-w-2xl sm:max-w-3xl`, element grid same as progress page (`gap-1.5 sm:gap-2`, `p-1.5 text-[11px] sm:p-2 sm:text-xs`, `truncate`), select `py-2.5`.

- [ ] **Step 2: Commit**

```bash
git add src/frontend/src/app/\(app\)/students/\[id\]/page.tsx
git commit -m "style(frontend): responsive student profile page"
```

---

## Task 10: Connections page — responsive width

**Files:**
- Modify: `src/frontend/src/app/(app)/connections/page.tsx`

- [ ] **Step 1: Update connections/page.tsx**

```tsx
"use client"

import { useState } from "react"
import { toast } from "sonner"
import { useInvite, useRelationships, usePendingInvites, useAcceptInvite, useEndRelationship } from "@/lib/api/relationships"

export default function ConnectionsPage() {
  const { data: rels } = useRelationships()
  const { data: pending } = usePendingInvites()
  const invite = useInvite()
  const acceptInvite = useAcceptInvite()
  const endRel = useEndRelationship()

  const [email, setEmail] = useState("")

  const handleInvite = async () => {
    if (!email) return
    try {
      await invite.mutateAsync({ skater_email: email })
      toast.success("Приглашение отправлено")
      setEmail("")
    } catch {
      toast.error("Ошибка отправки")
    }
  }

  const activeRels = (rels?.relationships ?? []).filter((r) => r.status === "active")

  return (
    <div className="mx-auto max-w-2xl space-y-6 sm:max-w-3xl">
      <h1 className="text-lg font-semibold">Связи</h1>

      <div className="space-y-2">
        <p className="text-sm font-medium">Пригласить ученика</p>
        <div className="flex gap-2">
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="email@example.com"
            className="flex-1 rounded-xl border border-border bg-background px-3 py-2.5 text-sm"
          />
          <button onClick={handleInvite} className="whitespace-nowrap rounded-xl bg-primary text-primary-foreground px-4 py-2.5 text-sm">
            Пригласить
          </button>
        </div>
      </div>

      {pending && pending.relationships.length > 0 && (
        <div className="space-y-2">
          <p className="text-sm font-medium">Входящие приглашения</p>
          {pending.relationships.map((r) => (
            <div key={r.id} className="flex items-center justify-between rounded-xl border border-border p-3">
              <span className="text-sm truncate mr-2">{r.coach_name ?? r.coach_id}</span>
              <button onClick={() => acceptInvite.mutateAsync(r.id)} className="shrink-0 rounded-lg bg-primary px-3 py-1.5 text-xs text-primary-foreground">
                Принять
              </button>
            </div>
          ))}
        </div>
      )}

      {activeRels.length > 0 && (
        <div className="space-y-2">
          <p className="text-sm font-medium">Активные связи</p>
          {activeRels.map((r) => (
            <div key={r.id} className="flex items-center justify-between rounded-xl border border-border p-3">
              <span className="text-sm truncate mr-2">{r.skater_name ?? r.skater_id}</span>
              <button onClick={() => endRel.mutateAsync(r.id)} className="shrink-0 text-xs text-muted-foreground hover:text-red-500">
                Завершить
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
```

Changes: `max-w-lg` → `max-w-2xl sm:max-w-3xl`, `py-2` → `py-2.5` on inputs for touch targets, `whitespace-nowrap` on invite button, `truncate mr-2` on name text to prevent overflow, `shrink-0` on action buttons.

- [ ] **Step 2: Commit**

```bash
git add src/frontend/src/app/\(app\)/connections/page.tsx
git commit -m "style(frontend): responsive connections page"
```

---

## Task 11: Profile page — responsive form grid

**Files:**
- Modify: `src/frontend/src/app/(app)/profile/page.tsx`

- [ ] **Step 1: Update profile/page.tsx**

```tsx
"use client"

import { useRouter } from "next/navigation"
import { type FormEvent, useState } from "react"
import { toast } from "sonner"
import { useAuth } from "@/components/auth-provider"
import { FormField, FormTextarea } from "@/components/form-field"
import { Button } from "@/components/ui/button"
import { useTranslations } from "@/i18n"

export default function ProfilePage() {
  const { user, isLoading, logout } = useAuth()
  const router = useRouter()
  const t = useTranslations("profile")
  const tc = useTranslations("common")

  const [displayName, setDisplayName] = useState("")
  const [bio, setBio] = useState("")
  const [height, setHeight] = useState("")
  const [weight, setWeight] = useState("")
  const [saving, setSaving] = useState(false)

  if (isLoading || !user) return <div className="text-center text-muted-foreground">{tc("loading")}</div>

  async function handleSave(e: FormEvent) {
    e.preventDefault()
    setSaving(true)
    try {
      const { updateProfile } = await import("@/lib/auth")
      await updateProfile({
        display_name: displayName || undefined,
        bio: bio || undefined,
        height_cm: height ? Number.parseInt(height, 10) : undefined,
        weight_kg: weight ? Number.parseFloat(weight) : undefined,
      })
      toast.success(t("updateSuccess"))
    } catch {
      toast.error(t("updateError"))
    } finally {
      setSaving(false)
    }
  }

  async function handleLogout() {
    await logout()
    document.cookie = "sb_auth=; path=/; max-age=0"
    router.push("/login")
  }

  return (
    <div className="mx-auto max-w-lg space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="nike-h1">{t("title")}</h1>
        <button
          type="button"
          onClick={handleLogout}
          className="text-sm text-muted-foreground hover:text-foreground"
        >
          {t("signOut")}
        </button>
      </div>

      <form onSubmit={handleSave} className="space-y-4">
        <FormField label="Email" id="email" type="email" value={user.email} disabled />
        <FormField
          label={t("name")}
          id="name"
          type="text"
          value={displayName}
          onChange={e => setDisplayName(e.target.value)}
        />
        <FormTextarea
          label={t("bio")}
          id="bio"
          value={bio}
          onChange={e => setBio(e.target.value)}
          rows={3}
        />
        <div className="grid grid-cols-2 gap-3 sm:gap-4">
          <FormField
            label={t("height")}
            id="height"
            type="number"
            value={height}
            onChange={e => setHeight(e.target.value)}
            min={50}
            max={250}
          />
          <FormField
            label={t("weight")}
            id="weight"
            type="number"
            value={weight}
            onChange={e => setWeight(e.target.value)}
            min={20}
            max={300}
            step={0.1}
          />
        </div>
        <Button type="submit" disabled={saving}>
          {saving ? tc("saving") : tc("save")}
        </Button>
      </form>
    </div>
  )
}
```

Changes: `gap-4` → `gap-3 sm:gap-4` on height/weight grid.

- [ ] **Step 2: Commit**

```bash
git add src/frontend/src/app/\(app\)/profile/page.tsx
git commit -m "style(frontend): responsive profile form grid"
```

---

## Task 12: Element picker — responsive grid

**Files:**
- Modify: `src/frontend/src/components/upload/element-picker.tsx`

- [ ] **Step 1: Update element-picker.tsx**

```tsx
"use client"

const ELEMENTS = [
  { id: "three_turn", label: "Тройка" },
  { id: "waltz_jump", label: "Вальсовый" },
  { id: "toe_loop", label: "Перекидной" },
  { id: "flip", label: "Флип" },
  { id: "salchow", label: "Сальхов" },
  { id: "loop", label: "Петля" },
  { id: "lutz", label: "Лютц" },
  { id: "axel", label: "Аксель" },
]

export function ElementPicker({ value, onChange }: { value: string | null; onChange: (id: string) => void }) {
  return (
    <div className="grid grid-cols-3 gap-2 sm:grid-cols-4">
      {ELEMENTS.map((el) => (
        <button
          key={el.id}
          onClick={() => onChange(el.id)}
          className={`truncate rounded-xl border px-2 py-2.5 text-center text-xs sm:p-3 sm:text-sm ${
            value === el.id ? "border-primary bg-primary/10 text-primary" : "border-border hover:bg-accent/50"
          }`}
        >
          {el.label}
        </button>
      ))}
    </div>
  )
}
```

No changes needed — already responsive from previous fix (`grid-cols-3 sm:grid-cols-4`, `text-xs sm:text-sm`, `truncate`).

- [ ] **Step 2: Skip commit (no changes)**

---

## Task 13: Trend chart — responsive height

**Files:**
- Modify: `src/frontend/src/components/progress/trend-chart.tsx`

- [ ] **Step 1: Update trend-chart.tsx**

```tsx
"use client"

import { ResponsiveContainer, LineChart, Line, ReferenceArea, XAxis, YAxis } from "recharts"
import type { TrendResponse } from "@/types"

const TREND_LABELS: Record<string, string> = { improving: "Улучшение", stable: "Стабильно", declining: "Ухудшение" }

export function TrendChart({ data }: { data: TrendResponse }) {
  if (!data.data_points.length) {
    return <p className="text-center text-muted-foreground py-10">Нет данных</p>
  }

  const refMin = data.reference_range?.min
  const refMax = data.reference_range?.max

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm">
        <span>{data.metric_name}</span>
        <span className={data.trend === "improving" ? "text-green-500" : data.trend === "declining" ? "text-red-500" : "text-muted-foreground"}>
          {TREND_LABELS[data.trend]}
        </span>
      </div>
      <div className="h-[200px] sm:h-[250px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data.data_points} margin={{ top: 10, right: 10, bottom: 0, left: -10 }}>
            {refMin !== undefined && refMax !== undefined && (
              <ReferenceArea y1={refMin} y2={refMax} fill="#22c55e" fillOpacity={0.1} ifOverflow="extendDomain" />
            )}
            <XAxis dataKey="date" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Line type="monotone" dataKey="value" stroke="hsl(var(--primary))" strokeWidth={2} dot={{ r: 4 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      {data.current_pr !== null && (
        <p className="text-sm text-amber-500 font-medium">PR: {data.current_pr.toFixed(3)}</p>
      )}
    </div>
  )
}
```

Changes: wrapped chart in `div` with `h-[200px] sm:h-[250px]`, `ResponsiveContainer` uses `height="100%"`. Shorter on mobile, taller on desktop.

- [ ] **Step 2: Commit**

```bash
git add src/frontend/src/components/progress/trend-chart.tsx
git commit -m "style(frontend): responsive trend chart height"
```

---

## Task 14: Session card — responsive padding

**Files:**
- Modify: `src/frontend/src/components/session/session-card.tsx`

- [ ] **Step 1: Update session-card.tsx**

```tsx
"use client"

import Link from "next/link"
import { Award, Clock, Loader2 } from "lucide-react"
import type { Session } from "@/types"

const ELEMENT_NAMES: Record<string, string> = {
  three_turn: "Тройка", waltz_jump: "Вальсовый", toe_loop: "Перекидной",
  flip: "Флип", salchow: "Сальхов", loop: "Петля",
  lutz: "Лютц", axel: "Аксель",
}

function relativeTime(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return "только что"
  if (mins < 60) return `${mins} мин назад`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `${hours} ч назад`
  const days = Math.floor(hours / 24)
  return `${days} дн назад`
}

function scoreColor(score: number | null): string {
  if (score === null) return "text-muted-foreground"
  if (score >= 0.8) return "text-green-500"
  if (score >= 0.5) return "text-amber-500"
  return "text-red-500"
}

export function SessionCard({ session }: { session: Session }) {
  const hasPR = session.metrics.some((m) => m.is_pr)

  return (
    <Link href={`/sessions/${session.id}`} className="block">
      <div className="rounded-2xl border border-border p-3 sm:p-4 hover:bg-accent/30 transition-colors">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <p className="font-medium truncate">{ELEMENT_NAMES[session.element_type] ?? session.element_type}</p>
            <p className="text-xs text-muted-foreground flex items-center gap-1">
              <Clock className="h-3 w-3 shrink-0" />
              {relativeTime(session.created_at)}
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            {hasPR && <Award className="h-4 w-4 text-amber-500" />}
            {session.overall_score !== null && (
              <span className={`text-sm font-medium ${scoreColor(session.overall_score)}`}>
                {Math.round(session.overall_score * 100)}%
              </span>
            )}
          </div>
        </div>

        {session.status !== "done" ? (
          <div className="mt-2 flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-3 w-3 animate-spin" />
            {session.status === "processing" ? "Анализ..." : "Загрузка..."}
          </div>
        ) : (
          <div className="mt-2 flex flex-wrap gap-x-3 gap-y-0.5 text-xs text-muted-foreground">
            {session.metrics.slice(0, 3).map((m) => (
              <span key={m.metric_name}>{m.metric_name}: {m.metric_value.toFixed(2)}</span>
            ))}
          </div>
        )}
      </div>
    </Link>
  )
}
```

Changes: `p-4` → `p-3 sm:p-4`, `gap-2` on top row for overflow safety, `min-w-0 truncate` on element name, `shrink-0` on score area, `flex-wrap gap-x-3 gap-y-0.5` on metrics (was `flex gap-3`), show up to 3 metrics (was 2).

- [ ] **Step 2: Commit**

```bash
git add src/frontend/src/components/session/session-card.tsx
git commit -m "style(frontend): responsive session card"
```

---

## Self-Review

**Spec coverage:**
- iPhone (375px): responsive grids, touch targets, safe areas ✅
- iPad (768px): scaled content widths, visible bottom dock ✅
- MacBook (1440px): max-w-5xl container, desktop nav tabs ✅

**Placeholder scan:** None found. All code is complete.

**Type consistency:** No type changes. Only Tailwind class modifications.

**Key responsive patterns applied:**
- Content: `max-w-2xl sm:max-w-3xl` (pages), `max-w-5xl` (layout container)
- Element grids: `grid-cols-4 gap-1.5 sm:gap-2` with `truncate text-[11px] sm:text-xs`
- Cards: `p-3 sm:p-4`
- Touch targets: `py-2.5` on inputs/buttons (44px+), bottom dock `px-4 py-1.5`
- Safe areas: `env(safe-area-inset-bottom)` on bottom dock, `env(safe-area-inset-top)` on auth header
- Mobile viewport: `min-h-[dvh]` on auth layout
- Chart height: `h-[200px] sm:h-[250px]`
- Overflow prevention: `truncate`, `min-w-0`, `shrink-0`, `flex-wrap`
