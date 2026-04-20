# Header Nav Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the triple navigation (header AppNav + sidebar + bottom-tabs) with a single horizontal tab bar in the header, styled like Nike Run Club / Strava mobile app.

**Architecture:** Single `AppNav` component renders all tabs inline. On mobile, tabs scroll horizontally with `overflow-x-auto` and `scrollbar-hide`. On desktop, tabs display with text labels. Coach-only tabs (Ученики) conditionally shown based on `hasStudents`. Remove `BottomTabs`, `Sidebar`, and `AppShell` wrapper entirely.

**Tech Stack:** Next.js 16, Tailwind CSS, lucide-react icons, React Query

---

## File Structure

### Delete
| File | Reason |
|------|--------|
| `src/frontend/src/components/layout/bottom-tabs.tsx` | Replaced by header tabs |
| `src/frontend/src/components/layout/sidebar.tsx` | Replaced by header tabs |
| `src/frontend/src/components/layout/app-shell.tsx` | No longer needed — hasStudents check moves to AppNav |

### Modify
| File | Change |
|------|--------|
| `src/frontend/src/components/app-nav.tsx` | Rewrite: horizontal tab bar with all nav items |
| `src/frontend/src/app/layout.tsx` | Remove max-w-7xl constraint from header, adjust header height |

### Modify
| File | Change |
|------|--------|
| `src/frontend/src/app/(app)/layout.tsx` | Remove AppShell wrapper, pass children directly |

---

## Nav Items (consolidated)

**All users:**
| Tab | href | Icon |
|-----|------|------|
| Лента | `/feed` | `Newspaper` |
| Запись | `/upload` | `Camera` |
| Прогресс | `/progress` | `BarChart3` |

**Coach only (hasStudents):**
| Tab | href | Icon |
|-----|------|------|
| Ученики | `/dashboard` | `Users` |

**Right side (not tabs):**
| Item | Action | Icon |
|------|--------|------|
| Profile | Link `/profile` | `User` |
| Logout | Button | `LogOut` |
| Theme | Toggle | `Sun/Moon` |

---

## Task 1: Rewrite AppNav as horizontal tab bar

**Files:**
- Modify: `src/frontend/src/components/app-nav.tsx`

- [ ] **Step 1: Rewrite app-nav.tsx**

```tsx
"use client"

import { BarChart3, Camera, LogIn, LogOut, Newspaper, Settings, User, Users } from "lucide-react"
import Link from "next/link"
import { usePathname, useRouter } from "next/navigation"
import { useQuery } from "@tanstack/react-query"
import { z } from "zod"
import { useAuth } from "@/components/auth-provider"
import { ThemeToggle } from "@/components/theme-toggle"
import { apiFetch } from "@/lib/api-client"

const RelationshipListSchema = z.object({
  relationships: z.array(z.object({ status: z.string() })),
})

export function AppNav() {
  const pathname = usePathname()
  const router = useRouter()
  const { isAuthenticated, user, logout } = useAuth()

  const { data: relsData } = useQuery({
    queryKey: ["relationships"],
    queryFn: () => apiFetch("/relationships", RelationshipListSchema),
    enabled: isAuthenticated,
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

  async function handleLogout() {
    await logout()
    router.push("/")
  }

  return (
    <nav className="flex items-center gap-0.5 overflow-x-auto scrollbar-hide">
      {tabs.map((tab) => {
        const Icon = tab.icon
        const active = isActive(tab.href)
        return (
          <Link
            key={tab.href}
            href={tab.href}
            className={`flex shrink-0 items-center gap-1.5 px-3 py-2 text-sm font-medium transition-colors whitespace-nowrap ${
              active
                ? "text-foreground"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <Icon className="h-4 w-4" />
            <span>{tab.label}</span>
          </Link>
        )
      })}

      <div className="ml-auto flex shrink-0 items-center gap-1">
        <ThemeToggle />
        {isAuthenticated ? (
          <>
            <Link
              href="/profile"
              className={`flex items-center gap-1.5 px-2 py-2 text-sm transition-colors hover:text-foreground ${
                isActive("/profile") ? "text-foreground" : "text-muted-foreground"
              }`}
            >
              <User className="h-4 w-4" />
            </Link>
            <button
              type="button"
              onClick={handleLogout}
              className="flex items-center p-2 text-muted-foreground transition-colors hover:text-foreground"
            >
              <LogOut className="h-4 w-4" />
            </button>
          </>
        ) : (
          <Link
            href="/login"
            className="flex items-center px-2 py-2 text-sm text-muted-foreground transition-colors hover:text-foreground"
          >
            <LogIn className="h-4 w-4" />
          </Link>
        )}
      </div>
    </nav>
  )
}
```

- [ ] **Step 2: Commit**

```bash
git add src/frontend/src/components/app-nav.tsx
git commit -m "refactor(frontend): rewrite AppNav as horizontal tab bar"
```

---

## Task 2: Remove BottomTabs, Sidebar, AppShell

**Files:**
- Delete: `src/frontend/src/components/layout/bottom-tabs.tsx`
- Delete: `src/frontend/src/components/layout/sidebar.tsx`
- Delete: `src/frontend/src/components/layout/app-shell.tsx`

- [ ] **Step 1: Remove layout components**

```bash
rm src/frontend/src/components/layout/bottom-tabs.tsx src/frontend/src/components/layout/sidebar.tsx src/frontend/src/components/layout/app-shell.tsx
```

- [ ] **Step 2: Commit**

```bash
git add -A
git commit -m "refactor(frontend): remove bottom-tabs, sidebar, app-shell components"
```

---

## Task 3: Simplify (app) layout and root header

**Files:**
- Modify: `src/frontend/src/app/(app)/layout.tsx`
- Modify: `src/frontend/src/app/layout.tsx`

- [ ] **Step 1: Simplify (app)/layout.tsx**

Replace content with a simple passthrough (AppShell is gone):

```tsx
export default function AppLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>
}
```

- [ ] **Step 2: Adjust root layout header**

In `src/frontend/src/app/layout.tsx`, remove the `max-w-7xl` constraint from the header so tabs can use full width. Change the header div to:

```tsx
<header className="sticky top-0 z-50 border-b border-border bg-background">
  <div className="flex h-[52px] items-center justify-between px-4">
```

Note: reduced height from 60px to 52px for a more compact app-like feel.

- [ ] **Step 3: Commit**

```bash
git add src/frontend/src/app/\(app\)/layout.tsx src/frontend/src/app/layout.tsx
git commit -m "refactor(frontend): simplify app layout, adjust header height"
```

---

## Task 4: Add scrollbar-hide utility

**Files:**
- Modify: `src/frontend/src/app/globals.css`

- [ ] **Step 1: Add scrollbar-hide utility class**

Append to `src/frontend/src/app/globals.css`:

```css
/* Hide scrollbar but keep scroll functionality */
.scrollbar-hide {
  -ms-overflow-style: none;
  scrollbar-width: none;
}
.scrollbar-hide::-webkit-scrollbar {
  display: none;
}
```

- [ ] **Step 2: Commit**

```bash
git add src/frontend/src/app/globals.css
git commit -m "feat(frontend): add scrollbar-hide utility class"
```

---

## Self-Review

**Spec coverage:**
- Single horizontal tab bar in header ✅
- Coach-only conditional tabs ✅
- Mobile app-like feel (no footer, no bottom tabs) ✅
- Profile/theme/logout in header ✅

**Placeholder scan:** None found.

**Type consistency:** Tab hrefs match actual page routes. Icons imported from lucide-react.

**Mobile behavior:** `overflow-x-auto` with `scrollbar-hide` enables horizontal scroll on narrow screens. Tabs have `shrink-0` to prevent squishing. `whitespace-nowrap` prevents label wrapping.
