# Profile + Statistics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign profile page into a Strava-like athlete dashboard with personal stats, recent activity, and quick-edit profile form.

**Architecture:** Extend existing profile page with a stats summary section at the top (total sessions, PRs, activity streak), followed by a personal records grid grouped by element type. Keep the edit form below. All data comes from existing backend endpoints (`/metrics/prs`, `/sessions`, `/metrics/diagnostics`) — no new backend endpoints needed. Add one new frontend API hook `usePRs()`.

**Tech Stack:** React Query, Zod schemas, existing Lucide icons, Tailwind CSS

---

### Task 1: Add `usePRs()` hook and Zod schema

**Files:**
- Modify: `src/frontend/src/lib/api/metrics.ts`

- [ ] **Step 1: Add PRs schema and hook to metrics.ts**

Append after the existing `useMetricRegistry` function (after line 57):

```typescript
const PRSchema = z.object({
  element_type: z.string(),
  metric_name: z.string(),
  value: z.number(),
  session_id: z.string(),
})

const PRListSchema = z.object({ prs: z.array(PRSchema) })

export function usePRs(userId?: string, elementType?: string) {
  const params = new URLSearchParams()
  if (userId) params.set("user_id", userId)
  if (elementType) params.set("element_type", elementType)
  return useQuery({
    queryKey: ["prs", userId, elementType],
    queryFn: () => apiFetch("/metrics/prs?" + params.toString(), PRListSchema),
  })
}
```

- [ ] **Step 2: Verify no TypeScript errors**

Run: `cd src/frontend && bunx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add src/frontend/src/lib/api/metrics.ts
git commit -m "feat(frontend): add usePRs hook for personal records"
```

---

### Task 2: Add i18n keys for profile stats

**Files:**
- Modify: `src/frontend/messages/ru.json`
- Modify: `src/frontend/messages/en.json`

- [ ] **Step 1: Add Russian translations**

In `ru.json`, replace the entire `"profile"` section with:

```json
"profile": {
  "title": "Профиль",
  "signOut": "Выйти",
  "name": "Имя",
  "bio": "О себе",
  "height": "Рост (см)",
  "weight": "Вес (кг)",
  "updateSuccess": "Профиль обновлён",
  "updateError": "Ошибка сохранения",
  "totalSessions": "Тренировок",
  "totalPRs": "Рекордов",
  "editProfile": "Редактировать",
  "personalRecords": "Личные рекорды",
  "noRecords": "Пока нет записей. Запишите первое видео!",
  "noPRs": "Пока нет рекордов.",
  "newPR": "Новый рекорд!",
  "viewSession": "Открыть"
},
```

- [ ] **Step 2: Add English translations**

In `en.json`, replace the entire `"profile"` section with:

```json
"profile": {
  "title": "Profile",
  "signOut": "Sign Out",
  "name": "Name",
  "bio": "Bio",
  "height": "Height (cm)",
  "weight": "Weight (kg)",
  "updateSuccess": "Profile updated",
  "updateError": "Save error",
  "totalSessions": "Sessions",
  "totalPRs": "PRs",
  "editProfile": "Edit",
  "personalRecords": "Personal Records",
  "noRecords": "No sessions yet. Record your first video!",
  "noPRs": "No personal records yet.",
  "newPR": "New PR!",
  "viewSession": "View"
},
```

- [ ] **Step 3: Commit**

```bash
git add src/frontend/messages/ru.json src/frontend/messages/en.json
git commit -m "feat(frontend): add i18n keys for profile statistics"
```

---

### Task 3: Build StatsSummary component

**Files:**
- Create: `src/frontend/src/components/profile/stats-summary.tsx`

- [ ] **Step 1: Create StatsSummary component**

```tsx
"use client"

import { Trophy, Video } from "lucide-react"
import { usePRs, useSessions } from "@/lib/api"

type Props = {
  userId?: string
}

export function StatsSummary({ userId }: Props) {
  const { data: sessionsData } = useSessions(userId)
  const { data: prsData } = usePRs(userId)

  const totalSessions = sessionsData?.total ?? 0
  const totalPRs = prsData?.prs.length ?? 0

  return (
    <div className="grid grid-cols-2 gap-3">
      <div className="flex items-center gap-3 rounded-xl border border-border p-4">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-primary/10">
          <Video className="h-5 w-5 text-primary" />
        </div>
        <div>
          <p className="text-2xl font-bold">{totalSessions}</p>
          <p className="text-xs text-muted-foreground">Тренировок</p>
        </div>
      </div>
      <div className="flex items-center gap-3 rounded-xl border border-border p-4">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-amber-500/10">
          <Trophy className="h-5 w-5 text-amber-500" />
        </div>
        <div>
          <p className="text-2xl font-bold">{totalPRs}</p>
          <p className="text-xs text-muted-foreground">Рекордов</p>
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd src/frontend && bunx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add src/frontend/src/components/profile/stats-summary.tsx
git commit -m "feat(frontend): add StatsSummary component"
```

---

### Task 4: Build PersonalRecords component

**Files:**
- Create: `src/frontend/src/components/profile/personal-records.tsx`

- [ ] **Step 1: Create PersonalRecords component**

```tsx
"use client"

import { Trophy } from "lucide-react"
import Link from "next/link"
import { useMetricRegistry, usePRs } from "@/lib/api"

const ELEMENT_LABELS: Record<string, string> = {
  waltz_jump: "Вальсовый",
  toe_loop: "Перекидной",
  flip: "Флип",
  salchow: "Сальхов",
  loop: "Петля",
  lutz: "Лютц",
  axel: "Аксель",
  three_turn: "Тройка",
}

export function PersonalRecords({ userId }: { userId?: string }) {
  const { data: prsData } = usePRs(userId)
  const { data: registry } = useMetricRegistry()

  if (!prsData || !registry) return null

  const prs = prsData.prs
  if (prs.length === 0) {
    return (
      <div className="rounded-xl border border-border p-6 text-center text-sm text-muted-foreground">
        Пока нет рекордов.
      </div>
    )
  }

  // Group PRs by element type
  const grouped = prs.reduce<Record<string, typeof prs>>((acc, pr) => {
    if (!acc[pr.element_type]) acc[pr.element_type] = []
    acc[pr.element_type].push(pr)
    return acc
  }, {})

  return (
    <div className="space-y-4">
      {Object.entries(grouped).map(([elementType, elementPRs]) => (
        <div key={elementType}>
          <h3 className="mb-2 text-sm font-medium text-muted-foreground">
            {ELEMENT_LABELS[elementType] ?? elementType}
          </h3>
          <div className="space-y-1.5">
            {elementPRs.map((pr) => {
              const mdef = registry[pr.metric_name]
              if (!mdef) return null
              const formatted = pr.value.toFixed(Number((mdef as any).format?.replace(".", "")) || 2)
              return (
                <Link
                  key={`${pr.metric_name}-${pr.session_id}`}
                  href={`/sessions/${pr.session_id}`}
                  className="flex items-center justify-between rounded-lg border border-border px-3 py-2.5 transition-colors hover:bg-accent"
                >
                  <div className="flex items-center gap-2">
                    <Trophy className="h-4 w-4 text-amber-500" />
                    <span className="text-sm">{(mdef as any).label_ru ?? pr.metric_name}</span>
                  </div>
                  <span className="text-sm font-semibold">
                    {formatted} {(mdef as any).unit ?? ""}
                  </span>
                </Link>
              )
            })}
          </div>
        </div>
      ))}
    </div>
  )
}
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd src/frontend && bunx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add src/frontend/src/components/profile/personal-records.tsx
git commit -m "feat(frontend): add PersonalRecords component"
```

---

### Task 5: Build RecentActivity component

**Files:**
- Create: `src/frontend/src/components/profile/recent-activity.tsx`

- [ ] **Step 1: Create RecentActivity component**

```tsx
"use client"

import Link from "next/link"
import { useSessions } from "@/lib/api"

const ELEMENT_LABELS: Record<string, string> = {
  waltz_jump: "Вальсовый",
  toe_loop: "Перекидной",
  flip: "Флип",
  salchow: "Сальхов",
  loop: "Петля",
  lutz: "Лютц",
  axel: "Аксель",
  three_turn: "Тройка",
}

export function RecentActivity({ userId }: { userId?: string }) {
  const { data: sessionsData } = useSessions(userId)

  if (!sessionsData) return null

  const sessions = sessionsData.sessions.slice(0, 5)
  if (sessions.length === 0) {
    return (
      <div className="rounded-xl border border-border p-6 text-center text-sm text-muted-foreground">
        Пока нет записей. Запишите первое видео!
      </div>
    )
  }

  return (
    <div className="space-y-1.5">
      {sessions.map((s) => {
        const date = new Date(s.created_at).toLocaleDateString("ru-RU", {
          day: "numeric", month: "short",
        })
        const score = s.overall_score != null
          ? `${Math.round(s.overall_score * 100)}%`
          : null
        return (
          <Link
            key={s.id}
            href={`/sessions/${s.id}`}
            className="flex items-center justify-between rounded-lg border border-border px-3 py-2.5 transition-colors hover:bg-accent"
          >
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium">
                {ELEMENT_LABELS[s.element_type] ?? s.element_type}
              </p>
              <p className="text-xs text-muted-foreground">{date}</p>
            </div>
            {score && (
              <span className={`ml-2 text-sm font-semibold ${
                s.overall_score! >= 0.7 ? "text-green-500" : s.overall_score! >= 0.4 ? "text-amber-500" : "text-red-500"
              }`}>
                {score}
              </span>
            )}
          </Link>
        )
      })}
    </div>
  )
}
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd src/frontend && bunx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add src/frontend/src/components/profile/recent-activity.tsx
git commit -m "feat(frontend): add RecentActivity component"
```

---

### Task 6: Redesign profile page

**Files:**
- Modify: `src/frontend/src/app/(app)/profile/page.tsx`

- [ ] **Step 1: Replace profile page with new design**

Replace the entire contents of `src/frontend/src/app/(app)/profile/page.tsx` with:

```tsx
"use client"

import { Pencil } from "lucide-react"
import { useRouter } from "next/navigation"
import { type FormEvent, useState } from "react"
import { toast } from "sonner"
import { useAuth } from "@/components/auth-provider"
import { FormField, FormTextarea } from "@/components/form-field"
import { PersonalRecords } from "@/components/profile/personal-records"
import { RecentActivity } from "@/components/profile/recent-activity"
import { StatsSummary } from "@/components/profile/stats-summary"
import { Button } from "@/components/ui/button"
import { useTranslations } from "@/i18n"

export default function ProfilePage() {
  const { user, isLoading, logout } = useAuth()
  const router = useRouter()
  const t = useTranslations("profile")
  const tc = useTranslations("common")

  const [editing, setEditing] = useState(false)
  const [displayName, setDisplayName] = useState("")
  const [bio, setBio] = useState("")
  const [height, setHeight] = useState("")
  const [weight, setWeight] = useState("")
  const [saving, setSaving] = useState(false)

  if (isLoading) return <div className="text-center text-muted-foreground">{tc("loading")}</div>
  if (!user) return null

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
      setEditing(false)
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

  function startEditing() {
    setDisplayName(user.display_name ?? "")
    setBio(user.bio ?? "")
    setHeight(user.height_cm?.toString() ?? "")
    setWeight(user.weight_kg?.toString() ?? "")
    setEditing(true)
  }

  return (
    <div className="mx-auto max-w-lg space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="nike-h1">{t("title")}</h1>
        <div className="flex items-center gap-2">
          {!editing && (
            <button
              type="button"
              onClick={startEditing}
              className="flex items-center gap-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground"
            >
              <Pencil className="h-3.5 w-3.5" />
              {t("editProfile")}
            </button>
          )}
          <button
            type="button"
            onClick={handleLogout}
            className="text-sm text-muted-foreground transition-colors hover:text-foreground"
          >
            {t("signOut")}
          </button>
        </div>
      </div>

      {/* User info card */}
      <div className="rounded-xl border border-border p-4">
        <div className="flex items-center gap-3">
          <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-primary/10 text-lg font-bold text-primary">
            {(user.display_name ?? user.email)[0].toUpperCase()}
          </div>
          <div className="min-w-0 flex-1">
            <p className="truncate text-base font-semibold">
              {user.display_name ?? user.email}
            </p>
            {user.bio && (
              <p className="truncate text-sm text-muted-foreground">{user.bio}</p>
            )}
          </div>
        </div>
        {(user.height_cm || user.weight_kg) && (
          <div className="mt-3 flex gap-4 text-sm text-muted-foreground">
            {user.height_cm && <span>{user.height_cm} см</span>}
            {user.weight_kg && <span>{user.weight_kg} кг</span>}
          </div>
        )}
      </div>

      {/* Edit form (collapsible) */}
      {editing && (
        <form onSubmit={handleSave} className="space-y-4 rounded-xl border border-border p-4">
          <FormField label={t("name")} id="name" type="text" value={displayName} onChange={e => setDisplayName(e.target.value)} />
          <FormTextarea label={t("bio")} id="bio" value={bio} onChange={e => setBio(e.target.value)} rows={3} />
          <div className="grid grid-cols-2 gap-3 sm:gap-4">
            <FormField label={t("height")} id="height" type="number" value={height} onChange={e => setHeight(e.target.value)} min={50} max={250} />
            <FormField label={t("weight")} id="weight" type="number" value={weight} onChange={e => setWeight(e.target.value)} min={20} max={300} step={0.1} />
          </div>
          <div className="flex gap-2">
            <Button type="submit" disabled={saving}>
              {saving ? tc("saving") : tc("save")}
            </Button>
            <Button type="button" variant="ghost" onClick={() => setEditing(false)}>
              Отмена
            </Button>
          </div>
        </form>
      )}

      {/* Stats summary */}
      <StatsSummary />

      {/* Personal Records */}
      <div>
        <h2 className="mb-3 text-sm font-medium">{t("personalRecords")}</h2>
        <PersonalRecords />
      </div>

      {/* Recent Activity */}
      <div>
        <h2 className="mb-3 text-sm font-medium">Последние записи</h2>
        <RecentActivity />
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Verify the page renders**

Run: `cd src/frontend && bunx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add src/frontend/src/app/\(app\)/profile/page.tsx
git commit -m "feat(frontend): redesign profile page with stats, PRs, recent activity"
```

---

### Task 7: Export new hooks from barrel file

**Files:**
- Modify: `src/frontend/src/lib/api/index.ts`

- [ ] **Step 1: Add usePRs to barrel export**

Check if `src/frontend/src/lib/api/index.ts` exists. If it does, add `usePRs` to its exports. If not, this step is skipped (consumers import directly from `@/lib/api/metrics`).

Read the file first, then add the export if needed.

- [ ] **Step 2: Commit if changes were made**

```bash
git add src/frontend/src/lib/api/index.ts
git commit -m "feat(frontend): export usePRs from api barrel"
```

---

### Task 8: Visual QA and cleanup

**Files:**
- No new files (verification only)

- [ ] **Step 1: Run TypeScript check**

Run: `cd src/frontend && bunx tsc --noEmit`
Expected: No errors

- [ ] **Step 2: Run linter**

Run: `cd src/frontend && bunx next lint`
Expected: No errors or warnings related to new files

- [ ] **Step 3: Manual visual check**

Open `http://localhost:3000/profile` and verify:
1. User info card shows avatar initial, name, bio, height/weight
2. Stats summary shows total sessions and PRs
3. Personal records section groups by element type, each PR links to session
4. Recent activity shows last 5 sessions with score color coding
5. Edit button toggles the form, Cancel hides it
6. On mobile (resize to 375px): everything stacks properly, cards are full-width
7. On iPad (768px): layout is comfortable with max-w-lg

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix(frontend): profile page QA fixes"
```
