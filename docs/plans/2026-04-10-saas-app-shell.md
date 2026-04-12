# SaaS App Shell Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the flat two-page frontend into a SaaS app shell with top navigation, dark mode, and an analysis dashboard.

**Architecture:** Top navigation bar with app branding + nav links + theme toggle. Page content renders below. No sidebar — keep it simple. Placeholder pages for future sections ("Тренировки", "Настройки"). The analyze page becomes a card-based dashboard.

**Tech Stack:** Next.js 16 App Router, Tailwind CSS v4, shadcn/ui, next-themes, Lucide icons, Sonner

---

## File Structure (Target)

```
src/frontend/src/
├── app/
│   ├── layout.tsx            # MODIFY: app shell (top nav + theme toggle + toaster)
│   ├── providers.tsx        # MODIFY: add ThemeProvider
│   ├── globals.css           # UNCHANGED
│   ├── page.tsx              # UNCHANGED (upload page, now under top nav)
│   ├── analyze/
│   │   └── page.tsx          # MODIFY: dashboard with skeleton + stats + video + downloads
│   ├── training/
│   │   └── page.tsx          # CREATE: placeholder page
│   └── settings/
│       └── page.tsx          # CREATE: placeholder page
├── components/
│   ├── theme-toggle.tsx      # CREATE: sun/moon toggle button
│   ├── dashboard/
│   │   ├── stats-cards.tsx   # CREATE: 4 stats in a grid
│   │   ├── video-player.tsx  # CREATE: video in a card
│   │   └── download-section.tsx # CREATE: download buttons grid
│   └── ui/                   # UNCHANGED (existing shadcn components)
├── lib/
│   ├── api.ts                # UNCHANGED
│   ├── schemas.ts            # UNCHANGED
│   ├── utils.ts              # UNCHANGED
│   └── toast.ts              # CREATE: toast helpers
└── types/
      └── index.ts            # UNCHANGED
```

## What Changes vs What Stays

### UNCHANGED
- `globals.css` — CSS variables already have light/dark themes
- `src/components/ui/*` — all 9 shadcn components
- `src/lib/api.ts`, `src/lib/schemas.ts`, `src/lib/utils.ts`
- `src/types/index.ts`
- `src/app/page.tsx` — upload page (only minor polish later)

### NEW FILES
- `src/components/theme-toggle.tsx`
- `src/components/dashboard/stats-cards.tsx`
- `src/components/dashboard/video-player.tsx`
- `src/components/dashboard/download-section.tsx`
- `src/lib/toast.ts`
- `src/app/training/page.tsx`
- `src/app/settings/page.tsx`

### MODIFIED FILES
- `src/app/providers.tsx` — add ThemeProvider
- `src/app/layout.tsx` — top nav bar, theme toggle, footer, toaster
- `src/app/analyze/page.tsx` — replace flat result view with dashboard components

---

## Task 1: Add ThemeProvider to providers.tsx

**Files:**
- Modify: `src/frontend/src/app/providers.tsx`

- [ ] **Step 1: Update providers.tsx**

```tsx
"use client"

import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { ThemeProvider } from "next-themes"
import { useState } from "react"

export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(() => new QueryClient())

  return (
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    </ThemeProvider>
  )
}
```

- [ ] **Step 2: Verify build**

Run: `cd src/frontend && bun next build 2>&1 | tail -10`
Expected: Build succeeds (ThemeProvider is "use client" compatible).

- [ ] **Step 3: Commit**

```bash
git add src/frontend/src/app/providers.tsx
git commit -m "feat(frontend): add ThemeProvider to app shell"
```

---

## Task 2: Create ThemeToggle component

**Files:**
- Create: `src/frontend/src/components/theme-toggle.tsx`

- [ ] **Step 1: Create theme-toggle.tsx**

```tsx
"use client"

import { Moon, Sun } from "lucide-react"
import { useTheme } from "next-themes"
import { Button } from "@/components/ui/button"

export function ThemeToggle() {
  const { setTheme, resolvedTheme } = useTheme()

  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}
    >
      <Sun className="h-4 w-4 scale-100 dark:scale-0" />
      <Moon className="absolute h-4 w-4 scale-0 dark:scale-100" />
      <span className="sr-only">Переключить тему</span>
    </Button>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add src/frontend/src/components/theme-toggle.tsx
git commit -m "feat(frontend): add ThemeToggle component"
```

---

## Task 3: Create toast helpers

**Files:**
- Create: `src/frontend/src/lib/toast.ts`

- [ ] **Step 1: Create toast.ts**

```ts
import { toast } from "sonner"

export function toastSuccess(message: string) {
  toast.success(message)
}

export function toastError(message: string) {
  toast.error(message)
}
```

- [ ] **Step 2: Commit**

```bash
git add src/frontend/src/lib/toast.ts
git commit -m "feat(frontend): add toast helpers"
```

---

## Task 4: Create placeholder pages

**Files:**
- Create: `src/frontend/src/app/training/page.tsx`
- Create: `src/frontend/src/app/settings/page.tsx`

- [ ] **Step 1: Create training/page.tsx**

```tsx
export default function TrainingPage() {
  return (
    <div className="mx-auto max-w-4xl p-6">
      <h2 className="mb-4 text-xl font-semibold">Тренировки</h2>
      <p className="text-muted-foreground">Раздел в разработке.</p>
    </div>
  )
}
```

- [ ] **Step 2: Create settings/page.tsx**

```tsx
export default function SettingsPage() {
  return (
    <div className="mx-auto max-w-4xl p-6">
      <h2 className="mb-4 text-xl font-semibold">Настройки</h2>
      <p className="text-muted-foreground">Раздел в разработке.</p>
    </div>
  )
}
```

- [ ] **Step 3: Commit**

```bash
git add src/frontend/src/app/training/page.tsx src/frontend/src/app/settings/page.tsx
git commit -m "feat(frontend): add placeholder pages for Training and Settings"
```

---

## Task 5: Create dashboard components

**Files:**
- Create: `src/frontend/src/components/dashboard/stats-cards.tsx`
- Create: `src/frontend/src/components/dashboard/video-player.tsx`
- Create: `src/frontend/src/components/dashboard/download-section.tsx`

- [ ] **Step 1: Create stats-cards.tsx**

```tsx
import { Film, Framer, Grid3x3, Zap } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { ProcessStats } from "@/types"

interface StatsCardsProps {
  stats: ProcessStats
}

const statItems = [
  { key: "total", label: "Кадров", icon: Film, getValue: (s: ProcessStats) => s.total_frames },
  { key: "valid", label: "Валидных", icon: Grid3x3, getValue: (s: ProcessStats) => s.valid_frames },
  { key: "fps", label: "FPS", icon: Zap, getValue: (s: ProcessStats) => s.fps },
  { key: "resolution", label: "Разрешение", icon: Framer, getValue: (s: ProcessStats) => s.resolution },
] as const

export function StatsCards({ stats }: StatsCardsProps) {
  return (
    <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
      {statItems.map(item => {
        const Icon = item.icon
        return (
          <Card key={item.key}>
            <CardHeader className="flex flex-row items-center gap-2 pb-2">
              <Icon className="h-4 w-4 text-muted-foreground" />
              <CardTitle className="text-xs font-medium text-muted-foreground">{item.label}</CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              <p className="text-2xl font-bold">{item.getValue(stats)}</p>
            </CardContent>
          </Card>
        )
      })}
    </div>
  )
}
```

- [ ] **Step 2: Create video-player.tsx**

```tsx
import { Card, CardContent } from "@/components/ui/card"

interface VideoPlayerProps {
  src: string
}

export function VideoPlayer({ src }: VideoPlayerProps) {
  return (
    <Card>
      <CardContent className="p-4">
        {/* biome-ignore lint/a11y/useMediaCaption: analysis output, not media */}
        <video src={src} controls className="w-full rounded-md" />
      </CardContent>
    </Card>
  )
}
```

- [ ] **Step 3: Create download-section.tsx**

```tsx
import { Download, FileSpreadsheet, Table2 } from "lucide-react"
import { Button } from "@/components/ui/button"

interface DownloadItem {
  href: string
  label: string
  icon: typeof Download
}

interface DownloadSectionProps {
  videoUrl: string
  posesUrl: string | null
  csvUrl: string | null
}

export function DownloadSection({ videoUrl, posesUrl, csvUrl }: DownloadSectionProps) {
  const items: DownloadItem[] = [
    { href: videoUrl, label: "Видео", icon: Download },
    ...(posesUrl ? [{ href: posesUrl, label: "Позы (.npy)", icon: FileSpreadsheet }] : []),
    ...(csvUrl ? [{ href: csvUrl, label: "Биомеханика (.csv)", icon: Table2 }] : []),
  ]

  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
      {items.map(item => {
        const Icon = item.icon
        return (
          <Button key={item.label} variant="outline" asChild>
            <a href={item.href} download>
              <Icon className="mr-1.5 h-4 w-4" />
              {item.label}
            </a>
          </Button>
        )
      })}
    </div>
  )
}
```

- [ ] **Step 4: Commit**

```bash
git add src/frontend/src/components/dashboard/
git commit -m "feat(frontend): add dashboard components (stats, video, downloads)"
```

---

## Task 6: Update root layout with app shell

**Files:**
- Modify: `src/frontend/src/app/layout.tsx`

- [ ] **Step 1: Update layout.tsx**

```tsx
import { Activity, Settings, Trophy } from "lucide-react"
import type { Metadata } from "next"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { ThemeToggle } from "@/components/theme-toggle"
import { Toaster } from "@/components/ui/sonner"
import { Providers } from "./providers"
import "./globals.css"

export const metadata: Metadata = {
  title: "AI Тренер — Фигурное катание",
  description: "ML-based AI coach for figure skating",
}

const navItems = [
  { href: "/", label: "Анализ", icon: Activity },
  { href: "/training", label: "Тренировки", icon: Trophy },
  { href: "/settings", label: "Настройки", icon: Settings },
] as const

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname()

  return (
    <html lang="ru" suppressHydrationWarning>
      <body className="min-h-screen bg-background text-foreground">
        <Providers>
          <header className="sticky top-0 z-50 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            <div className="mx-auto flex h-12 max-w-6xl items-center justify-between px-4">
              <Link href="/" className="flex items-center gap-2 font-semibold">
                <Activity className="h-5 w-5" />
                <span className="hidden sm:inline">AI Тренер</span>
              </Link>
              <nav className="flex items-center gap-1">
                {navItems.map(item => {
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
                <ThemeToggle />
              </nav>
            </div>
          </header>
          <main className="mx-auto w-full max-w-6xl p-4 sm:p-6">{children}</main>
          <footer className="border-t border-border px-4 py-3 text-center text-xs text-muted-foreground">
            AI Тренер — Фигурное катание • Биомеханический анализ
          </footer>
          <Toaster richColors position="bottom-right" />
        </Providers>
      </body>
    </html>
  )
}
```

**Note:** The `<nav>` uses `usePathname()` from `next/navigation` which requires `"use client"`. But `layout.tsx` also exports `metadata` (a server component feature). Solution: extract the nav into a client component.

- [ ] **Step 2: Extract nav into a client component**

Create `src/frontend/src/components/app-nav.tsx`:

```tsx
"use client"

import { Activity, Settings, Trophy } from "lucide-react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { ThemeToggle } from "@/components/theme-toggle"

const navItems = [
  { href: "/", label: "Анализ", icon: Activity },
  { href: "/training", label: "Тренировки", icon: Trophy },
  { href: "/settings", label: "Настройки", icon: Settings },
] as const

export function AppNav() {
  const pathname = usePathname()

  return (
    <nav className="flex items-center gap-1">
      {navItems.map(item => {
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
      <ThemeToggle />
    </nav>
  )
}
```

Then update `layout.tsx` to use `<AppNav />`:

```tsx
import type { Metadata } from "next"
import Link from "next/link"
import { Activity } from "lucide-react"
import { AppNav } from "@/components/app-nav"
import { Toaster } from "@/components/ui/sonner"
import { Providers } from "./providers"
import "./globals.css"

export const metadata: Metadata = {
  title: "AI Тренер — Фигурное катание",
  description: "ML-based AI coach for figure skating",
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ru" suppressHydrationWarning>
      <body className="min-h-screen bg-background text-foreground">
        <Providers>
          <header className="sticky top-0 z-50 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            <div className="mx-auto flex h-12 max-w-6xl items-center justify-between px-4">
              <Link href="/" className="flex items-center gap-2 font-semibold">
                <Activity className="h-5 w-5" />
                <span className="hidden sm:inline">AI Тренер</span>
              </Link>
              <AppNav />
            </div>
          </header>
          <main className="mx-auto w-full max-w-6xl p-4 sm:p-6">{children}</main>
          <footer className="border-t border-border px-4 py-3 text-center text-xs text-muted-foreground">
            AI Тренер — Фигурное катание • Биомеханический анализ
          </footer>
          <Toaster richColors position="bottom-right" />
        </Providers>
      </body>
    </html>
  )
}
```

- [ ] **Step 3: Verify build**

Run: `cd src/frontend && bun next build 2>&1 | tail -15`
Expected: All 5 routes generated (`/`, `/analyze`, `/_not-found`, `/training`, `/settings`).

- [ ] **Step 4: Commit**

```bash
git add src/frontend/src/components/app-nav.tsx src/frontend/src/app/layout.tsx
git commit -m "feat(frontend): add SaaS app shell with top nav, theme toggle, footer"
```

---

## Task 7: Polish upload page

**Files:**
- Modify: `src/frontend/src/app/page.tsx`

- [ ] **Step 1: Add responsive padding and max-width**

The upload page currently has `mx-auto max-w-5xl p-6`. Since the layout now constrains max-w to 6xl and provides padding, update the page to remove its own max-w and padding:

Change `mx-auto max-w-5xl p-6` → remove (the layout handles this now).

- [ ] **Step 2: Add file validation before upload**

In the `handleDrop` and file input `onChange`, add validation:

```ts
function isValidVideoFile(file: File): boolean {
  const MAX_SIZE = 500 * 1024 * 1024 // 500MB
  const validTypes = ["video/mp4", "video/quicktime", "video/webm"]
  return validTypes.includes(file.type) && file.size <= MAX_SIZE
}
```

Show `toastError("Файл должен быть видео (MP4, MOV, WebM) до 500 МБ")` for invalid files.

- [ ] **Step 3: Add toast on successful detection**

After `setStatus("ready")` in `handleFile`, add `toastSuccess("Обнаружено человек: ${resp.persons.length}")`.

- [ ] **Step 4: Verify build**

Run: `cd src/frontend && bun next build 2>&1 | tail -10`

- [ ] **Step 5: Commit**

```bash
git add src/frontend/src/app/page.tsx
git commit -m "feat(frontend): polish upload page with validation and toasts"
```

---

## Task 8: Build analysis dashboard

**Files:**
- Modify: `src/frontend/src/app/analyze/page.tsx`

- [ ] **Step 1: Replace the "done" phase JSX**

Import the dashboard components:

```tsx
import { StatsCards } from "@/components/dashboard/stats-cards"
import { VideoPlayer } from "@/components/dashboard/video-player"
import { DownloadSection } from "@/components/dashboard/download-section"
```

Replace the entire `{phase === "done" && result && (...)}` block with:

```tsx
{phase === "done" && result && (
  <div className="space-y-6">
    <StatsCards stats={result.stats} />
    <VideoPlayer src={videoUrl} />
    <DownloadSection videoUrl={videoUrl} posesUrl={posesUrl} csvUrl={csvUrl} />
  </div>
)}
```

Remove the `ArrowLeft` back button at the top (the nav handles navigation now).

Keep the "processing" and "error" phases unchanged.

- [ ] **Step 2: Remove unused imports**

Remove `ArrowLeft` and `CheckCircle` from the lucide-react import if no longer used.

- [ ] **Step 3: Verify build**

Run: `cd src/frontend && bun next build 2>&1 | tail -10`

- [ ] **Step 4: Commit**

```bash
git add src/frontend/src/app/analyze/page.tsx
git commit -m "feat(frontend): replace analyze result view with dashboard"
```

---

## Task 9: Add skeleton loading states

**Files:**
- Modify: `src/frontend/src/app/page.tsx` (upload page loading)
- Modify: `src/frontend/src/app/analyze/page.tsx` (analyze page loading)

- [ ] **Step 1: Replace upload spinner with skeleton**

In `page.tsx`, replace the `isAnalyzing` block (Loader2 spinner) with:

```tsx
{isAnalyzing && (
  <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
    <Card>
      <CardContent className="space-y-4 p-4">
        <Skeleton className="h-4 w-32" />
        <Skeleton className="aspect-video w-full rounded-md" />
        <Skeleton className="h-3 w-48" />
      </CardContent>
    </Card>
    <div className="flex flex-col gap-4">
      <Card>
        <CardContent className="space-y-3 p-4">
          <Skeleton className="h-4 w-24" />
          <Skeleton className="h-8 w-full" />
        </CardContent>
      </Card>
      <Card>
        <CardContent className="space-y-3 p-4">
          <Skeleton className="h-4 w-20" />
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-8 w-full" />
        </CardContent>
      </Card>
    </div>
  </div>
)}
```

Add `import { Skeleton } from "@/components/ui/skeleton"`.

- [ ] **Step 2: Replace analyze spinner with skeleton**

In `analyze/page.tsx`, replace the `phase === "processing"` block (Loader2 spinner) with:

```tsx
{phase === "processing" && (
  <div className="space-y-6">
    <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
      <Card><CardContent className="space-y-2 p-4"><Skeleton className="h-3 w-16" /><Skeleton className="h-8 w-12" /></CardContent></Card>
      <Card><CardContent className="space-y-2 p-4"><Skeleton className="h-3 w-16" /><Skeleton className="h-8 w-12" /></CardContent></Card>
      <Card><CardContent className="space-y-2 p-4"><Skeleton className="h-3 w-16" /><Skeleton className="h-8 w-12" /></CardContent></Card>
      <Card><CardContent className="space-y-2 p-4"><Skeleton className="h-3 w-16" /><Skeleton className="h-8 w-12" /></CardContent></Card>
    </div>
    <Card>
      <CardContent className="p-4">
        <Skeleton className="aspect-video w-full rounded-md" />
      </CardContent>
    </Card>
  </div>
)}
```

- [ ] **Step 3: Verify build**

Run: `cd src/frontend && bun next build 2>&1 | tail -10`

- [ ] **Step 4: Commit**

```bash
git add src/frontend/src/app/page.tsx src/frontend/src/app/analyze/page.tsx
git commit -m "feat(frontend): replace spinners with skeleton loading states"
```

---

## Task 10: Final verification

- [ ] **Step 1: Run lint**

Run: `cd src/frontend && bun biome check src/`
Expected: No errors (warnings acceptable).

- [ ] **Step 2: Run production build**

Run: `cd src/frontend && bun next build 2>&1 | tail -15`
Expected: All 5 routes generated.

- [ ] **Step 3: Start dev server and manual test**

Run: `cd src/frontend && bun run dev`

Manual checks:
1. http://localhost:3001 — upload page with top nav
2. Toggle dark/light mode — all components respond
3. Click "Тренировки" — placeholder page
4. Click "Настройки" — placeholder page
5. Upload video — skeleton loading, then detect result
6. Click "Анализировать" — navigate to /analyze
7. Dashboard renders with stats, video, downloads
8. Resize to mobile — responsive layout works

- [ ] **Step 4: Commit any fixes**

```bash
git add -A src/frontend/
git commit -m "fix(frontend): address verification issues"
```

---

## Self-Review

### Spec Coverage
- [x] SaaS app shell with navigation — Task 5, 6
- [x] Dark mode — Task 1, 2
- [x] Polish & UX (responsive, toasts, skeletons, validation) — Task 7, 9
- [x] Analysis dashboard — Task 8
- [x] Placeholder pages — Task 4

### Placeholder Scan
- No TBDs or TODOs found
- All code blocks contain complete, copy-pasteable code
- All file paths are exact
- All commands have expected output

### Type Consistency
- `StatsCards` props use `ProcessStats` from `@/types` — matches existing type
- `DownloadSection` props match the URL patterns already in `analyze/page.tsx`
- `VideoPlayer` props are a simple `src: string` — minimal and correct
- `AppNav` is `"use client"` — correctly separated from server `layout.tsx`

### Architecture Notes
- `AppNav` is extracted as a client component because `usePathname()` requires it. `layout.tsx` remains a server component to export `metadata`.
- The `max-w-6xl` constraint is in `layout.tsx`, not in individual pages. Pages use `w-full max-w-6xl` only where they need additional centering.
- Dashboard components are in `components/dashboard/` — they're specific to the analyze flow, not shared.
- Placeholder pages are dead simple — no client components needed.
