# frontend/CLAUDE.md — Next.js App

## Tech Stack

- **Framework**: Next.js 16 with Turbopack (middleware NOT supported — use server components for auth)
- **Language**: TypeScript, React 19
- **Styling**: Tailwind CSS v4, shadcn/ui, OKLCH color system
- **State**: React Query (@tanstack/react-query) + Zod schema validation
- **i18n**: next-intl (messages in `frontend/messages/ru.json`, `en.json`)
- **React Query**: @tanstack/react-query for server state
- **Icons**: Lucide React
- **Charts**: Recharts
- **Runtime**: bun (NOT npm/npx)

## Project Structure

```
frontend/
├── app/
│   ├── layout.tsx                    # Root layout (providers, toaster)
│   ├── page.tsx                      # Cookie-based redirect (feed or login)
│   ├── providers.tsx                 # ThemeProvider, QueryClientProvider
│   ├── (auth)/                       # Auth layout (redirects if logged in)
│   │   ├── login/page.tsx
│   │   └── register/page.tsx
│   └── (app)/                        # App layout (auth gate via cookies())
│       ├── layout.tsx                # Header + AppNav + BottomDock
│       ├── feed/page.tsx             # Session list (SessionCard)
│       ├── upload/page.tsx           # Camera recorder + file upload
│       ├── progress/page.tsx         # TrendChart + PeriodSelector
│       ├── dashboard/page.tsx        # Coach: student list
│       ├── profile/page.tsx          # Stats, PRs, recent activity, edit form
│       ├── sessions/[id]/page.tsx    # Session detail
│       ├── students/[id]/page.tsx    # Student profile (coach view)
│       ├── connections/page.tsx      # Coach-skater invites
│       └── settings/page.tsx         # Theme, language, timezone
├── components/
│   ├── app-nav.tsx                   # Desktop horizontal tabs + profile/logout
│   ├── auth-provider.tsx             # Token management, user state, refresh flow
│   ├── theme-toggle.tsx              # Light/dark/system
│   ├── form-field.tsx                # FormField, FormTextarea (with labels)
│   ├── layout/
│   │   └── bottom-dock.tsx           # Mobile+iPad bottom tab bar (lg:hidden)
│   ├── profile/
│   │   ├── stats-summary.tsx         # Total sessions + PRs cards
│   │   ├── personal-records.tsx      # PRs grouped by element type
│   │   └── recent-activity.tsx       # Last 5 sessions with scores
│   ├── progress/
│   │   ├── trend-chart.tsx            # Recharts LineChart with reference area
│   │   └── period-selector.tsx       # 7d/30d/90d/all toggle
│   ├── session/
│   │   └── session-card.tsx          # Session list item with score + metrics
│   ├── coach/
│   │   └── student-card.tsx          # Student list item for coach dashboard
│   ├── upload/
│   │   ├── camera-recorder.tsx        # MediaRecorder-based camera
│   │   └── chunked-uploader.tsx       # Multipart upload with progress
│   └── ui/                           # shadcn/ui components (button, slider, etc.)
├── lib/
│   ├── api-client.ts                 # apiFetch, apiPost, apiPatch, apiDelete, token helpers
│   ├── auth.ts                       # login, register, refreshToken, fetchMe, updateProfile
│   ├── api/
│   │   ├── sessions.ts              # useSessions, useSession, useCreateSession, etc.
│   │   ├── metrics.ts               # useTrend, useDiagnostics, useMetricRegistry, usePRs
│   │   └── connections.ts          # useConnections, useInviteConnection, etc.
│   ├── constants.ts                  # ELEMENT_TYPE_KEYS, ElementType
│   └── useMountEffect.ts            # Mount-only effect (no-use-effect skill)
├── i18n/
│   ├── index.ts                      # Re-exports useLocale, useTranslations from next-intl
│   ├── request.ts
│   └── actions.ts
├── types/
│   └── index.ts                      # Session, SessionMetric, TrendResponse, etc.
└── messages/
    ├── ru.json                       # Russian translations
    └── en.json                       # English translations
```

## React Patterns

### No useEffect (enforced)

**Never use `useEffect` directly.** Use these replacements:

| Instead of useEffect for... | Use |
|----------------------------|-----|
| Deriving state | Inline computation |
| Fetching data | `useQuery` / data-fetching libraries |
| Responding to user actions | Event handlers |
| One-time mount sync | `useMountEffect` from `@/lib/useMountEffect` |

### Data Fetching

Always use React Query with Zod validation:
```typescript
export function useSessions(userId?: string) {
  return useQuery({
    queryKey: ["sessions", userId],
    queryFn: () => apiFetch("/sessions?" + params, SessionListSchema),
  })
}
```

### Component Structure Convention

```typescript
export function Component({ prop }: Props) {
  // 1. Hooks first (useQuery, useAuth, useTranslations)
  // 2. Local state (useState)
  // 3. Computed values (NOT useEffect + setState)
  // 4. Event handlers
  // 5. Early returns (isLoading, !data)
  // 6. Render
}
```

## Auth Architecture

**No middleware** — Turbopack doesn't reliably support it. Instead:

1. **Server-side gate**: `(app)/layout.tsx` checks `sb_auth` cookie via `cookies()` + `redirect()`
2. **Auth layout**: `(auth)/layout.tsx` redirects authenticated users away from login/register
3. **Client state**: `auth-provider.tsx` provides `user`, `isLoading`, `logout` via React context
4. **Token refresh**: On mount, validates access token, falls back to refresh token, hard-redirects to `/login` on failure via `window.location.href`
5. **Cookie sync**: `setTokens()` sets `sb_auth=1` cookie, `clearTokens()` deletes it

## Responsive Design

- **Mobile first** (default = phone)
- **Breakpoints**: `sm:` (640px), `md:` (768px), `lg:` (1024px)
- **Bottom dock**: visible on iPhone + iPad (`lg:hidden`), hidden on MacBook
- **Desktop nav**: hidden on mobile (`hidden md:flex`), handles desktop tabs
- **Safe area**: `pb-[env(safe-area-inset-bottom)]` on bottom dock, `pt-[env(safe-area-inset-top)]` on auth layout
- **Viewport**: `min-h-[dvh]` for correct mobile height
- **Touch targets**: `px-4 py-1.5 rounded-lg` minimum on interactive elements

## Theming

- **Color space**: OKLCH (perceptually uniform)
- **CSS variables**: `--background`, `--foreground`, `--primary`, `--muted`, `--accent`, `--border`, `--score-good`, `--score-mid`, `--score-bad`, `--accent-gold`
- **Typography**: Inter Variable, `.nike-h1` (2rem/500), `.nike-h2` (1.5rem/500), `.nike-h3` (1.125rem/600), `.nike-body` (1rem/400)
- **Border radius**: `--radius-sm` (0.5rem), `--radius-md` (1.25rem), `--radius-lg` (1.875rem)
- **Dark mode**: Class-based via `next-themes` (`attribute="class"`)
- **No hard-coded colors** — use CSS custom properties via inline `style={{ color: "oklch(var(--score-good))" }}`

## Element Labels

Use i18n, not local constants:
```typescript
const te = useTranslations("elements")
const label = te("waltz_jump") // "Вальсовый" / "Waltz Jump"
```

## Before Committing

1. **TypeScript**: `bunx tsc --noEmit`
2. **Lint**: `bunx next lint`
