# Nike Design System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restyle the entire frontend to match Nike's monochrome, athletic aesthetic — flat design, pill buttons, Inter font (Cyrillic-capable), no shadows.

**Architecture:** Replace design tokens in globals.css (colors, radius, font), restyle shadcn/ui components via CSS variables + className tweaks, update page layouts and custom components to match Nike typography and spacing.

**Tech Stack:** Next.js 16, Tailwind CSS v4 (CSS-only config), shadcn/ui (radix-nova), @fontsource-variable/inter

---

### Task 1: Swap Font Package

**Files:**
- Modify: `src/frontend/package.json`

- [ ] **Step 1: Remove Geist, add Inter**

```bash
cd src/frontend
bun remove @fontsource-variable/geist
bun add @fontsource-variable/inter
```

- [ ] **Step 2: Verify install**

```bash
cd src/frontend && grep fontsource package.json
```

Expected: `"@fontsource-variable/inter": "^5.1"` (no geist)

- [ ] **Step 3: Commit**

```bash
git add src/frontend/package.json src/frontend/bun.lockb
git commit -m "chore(frontend): swap Geist font for Inter Variable (Cyrillic support)"
```

---

### Task 2: Rewrite globals.css Design Tokens

**Files:**
- Modify: `src/frontend/src/app/globals.css`

- [ ] **Step 1: Replace font import and @theme block**

Replace lines 1-25 with:

```css
@import "tailwindcss";
@import "tw-animate-css";
@import "shadcn/tailwind.css";
@import "@fontsource-variable/inter";

@custom-variant dark (&:is(.dark *));

@theme {
  --radius-sm: 0.5rem;
  --radius-md: 1.25rem;
  --radius-lg: 1.875rem;
  --color-background: oklch(1 0 0);
  --color-foreground: oklch(0.145 0 0);
  --color-muted: oklch(0.967 0 0);
  --color-muted-foreground: oklch(0.55 0 0);
  --color-border: oklch(0.815 0 0);
  --color-primary: oklch(0.145 0 0);
  --color-primary-foreground: oklch(1 0 0);
  --color-destructive: oklch(0.541 0.22 25);
  --color-destructive-foreground: oklch(0.98 0 0);
  --color-card: oklch(1 0 0);
  --color-card-foreground: oklch(0.145 0 0);
  --color-accent: oklch(0.905 0 0);
  --color-link: oklch(0.541 0.23 264);
}
```

- [ ] **Step 2: Replace body and @theme inline block**

Replace lines 27-75 (body + @theme inline) with:

```css
body {
  margin: 0;
  font-family: "Inter Variable", "Inter", "Helvetica Neue", Helvetica, Arial, sans-serif;
  background: var(--color-background);
  color: var(--color-foreground);
}

@theme inline {
  --font-heading: var(--font-sans);
  --font-sans: "Inter Variable", "Inter", "Helvetica Neue", Helvetica, Arial, sans-serif;
  --color-sidebar-ring: var(--sidebar-ring);
  --color-sidebar-border: var(--sidebar-border);
  --color-sidebar-accent-foreground: var(--sidebar-accent-foreground);
  --color-sidebar-accent: var(--sidebar-accent);
  --color-sidebar-primary-foreground: var(--sidebar-primary-foreground);
  --color-sidebar-primary: var(--sidebar-primary);
  --color-sidebar-foreground: var(--sidebar-foreground);
  --color-sidebar: var(--sidebar);
  --color-chart-5: var(--chart-5);
  --color-chart-4: var(--chart-4);
  --color-chart-3: var(--chart-3);
  --color-chart-2: var(--chart-2);
  --color-chart-1: var(--chart-1);
  --color-ring: var(--ring);
  --color-input: var(--input);
  --color-border: var(--border);
  --color-destructive: var(--destructive);
  --color-accent-foreground: var(--accent-foreground);
  --color-accent: var(--accent);
  --color-muted-foreground: var(--muted-foreground);
  --color-muted: var(--muted);
  --color-secondary-foreground: var(--secondary-foreground);
  --color-secondary: var(--secondary);
  --color-primary-foreground: var(--primary-foreground);
  --color-primary: var(--primary);
  --color-popover-foreground: var(--popover-foreground);
  --color-popover: var(--popover);
  --color-card-foreground: var(--card-foreground);
  --color-card: var(--card);
  --color-foreground: var(--foreground);
  --color-background: var(--background);
  --color-link: var(--link);
  --radius: 1.25rem;
  --radius-sm: calc(var(--radius) * 0.4);
  --radius-md: var(--radius);
  --radius-lg: calc(var(--radius) * 1.5);
  --radius-xl: calc(var(--radius) * 1.5);
  --radius-2xl: calc(var(--radius) * 1.8);
  --radius-3xl: calc(var(--radius) * 2.2);
  --radius-4xl: calc(var(--radius) * 2.6);
}
```

- [ ] **Step 3: Replace :root and .dark blocks**

Replace lines 77-144 (`:root` + `.dark`) with:

```css
:root {
  --background: oklch(1 0 0);
  --foreground: oklch(0.145 0 0);
  --card: oklch(1 0 0);
  --card-foreground: oklch(0.145 0 0);
  --popover: oklch(1 0 0);
  --popover-foreground: oklch(0.145 0 0);
  --primary: oklch(0.145 0 0);
  --primary-foreground: oklch(1 0 0);
  --secondary: oklch(0.967 0 0);
  --secondary-foreground: oklch(0.145 0 0);
  --muted: oklch(0.98 0 0);
  --muted-foreground: oklch(0.55 0 0);
  --accent: oklch(0.905 0 0);
  --accent-foreground: oklch(0.145 0 0);
  --destructive: oklch(0.541 0.22 25);
  --border: oklch(0.815 0 0);
  --input: oklch(0.815 0 0);
  --ring: oklch(0.48 0.19 264);
  --link: oklch(0.541 0.23 264);
  --chart-1: oklch(0.967 0 0);
  --chart-2: oklch(0.55 0 0);
  --chart-3: oklch(0.225 0 0);
  --chart-4: oklch(0.185 0 0);
  --chart-5: oklch(0.145 0 0);
  --radius: 1.25rem;
  --sidebar: oklch(1 0 0);
  --sidebar-foreground: oklch(0.145 0 0);
  --sidebar-primary: oklch(0.145 0 0);
  --sidebar-primary-foreground: oklch(1 0 0);
  --sidebar-accent: oklch(0.967 0 0);
  --sidebar-accent-foreground: oklch(0.145 0 0);
  --sidebar-border: oklch(0.815 0 0);
  --sidebar-ring: oklch(0.48 0.19 264);
}

.dark {
  --background: oklch(0.185 0 0);
  --foreground: oklch(0.98 0 0);
  --card: oklch(0.225 0 0);
  --card-foreground: oklch(0.98 0 0);
  --popover: oklch(0.225 0 0);
  --popover-foreground: oklch(0.98 0 0);
  --primary: oklch(0.98 0 0);
  --primary-foreground: oklch(0.145 0 0);
  --secondary: oklch(0.285 0 0);
  --secondary-foreground: oklch(0.98 0 0);
  --muted: oklch(0.285 0 0);
  --muted-foreground: oklch(0.68 0 0);
  --accent: oklch(0.285 0 0);
  --accent-foreground: oklch(0.98 0 0);
  --destructive: oklch(0.577 0.245 27);
  --border: oklch(0.145 0 0 / 15%);
  --input: oklch(0.145 0 0 / 20%);
  --ring: oklch(0.588 0.19 254);
  --link: oklch(0.588 0.19 254);
  --chart-1: oklch(0.967 0 0);
  --chart-2: oklch(0.55 0 0);
  --chart-3: oklch(0.225 0 0);
  --chart-4: oklch(0.185 0 0);
  --chart-5: oklch(0.145 0 0);
  --radius: 1.25rem;
  --sidebar: oklch(0.225 0 0);
  --sidebar-foreground: oklch(0.98 0 0);
  --sidebar-primary: oklch(0.98 0 0);
  --sidebar-primary-foreground: oklch(0.145 0 0);
  --sidebar-accent: oklch(0.285 0 0);
  --sidebar-accent-foreground: oklch(0.98 0 0);
  --sidebar-border: oklch(0.145 0 0 / 15%);
  --sidebar-ring: oklch(0.588 0.19 254);
}
```

- [ ] **Step 4: Add Nike typography utilities to @layer base**

Replace the existing `@layer base` block (lines 146-157) with:

```css
@layer base {
  * {
    @apply border-border outline-ring/50;
  }
  body {
    @apply bg-background text-foreground;
  }
  html {
    @apply font-sans;
  }
  .nike-display {
    font-size: clamp(3rem, 6vw, 6rem);
    font-weight: 500;
    line-height: 0.9;
    text-transform: uppercase;
    letter-spacing: -0.02em;
  }
  .nike-h1 {
    font-size: 2rem;
    font-weight: 500;
    line-height: 1.2;
  }
  .nike-h2 {
    font-size: 1.5rem;
    font-weight: 500;
    line-height: 1.2;
  }
  .nike-body {
    font-size: 1rem;
    font-weight: 400;
    line-height: 1.75;
  }
}
```

- [ ] **Step 5: Commit**

```bash
git add src/frontend/src/app/globals.css
git commit -m "refactor(frontend): Nike design tokens — colors, radius, typography, Inter font"
```

---

### Task 3: Restyle Button Component (Pill Shape + Flat)

**Files:**
- Modify: `src/frontend/src/components/ui/button.tsx`

- [ ] **Step 1: Update button base and variants**

In `buttonVariants` (line 8), replace the entire base class string:

**Old:** `"group/button inline-flex shrink-0 items-center justify-center rounded-lg border border-transparent bg-clip-padding text-sm font-medium whitespace-nowrap transition-all outline-none select-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 active:not-aria-[haspopup]:translate-y-px disabled:pointer-events-none disabled:opacity-50 aria-invalid:border-destructive aria-invalid:ring-3 aria-invalid:ring-destructive/20 dark:aria-invalid:border-destructive/50 dark:aria-invalid:ring-destructive/40 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4"`

**New:** `"group/button inline-flex shrink-0 items-center justify-center rounded-[1.875rem] border border-transparent bg-clip-padding text-sm font-medium whitespace-nowrap transition-colors duration-200 outline-none select-none focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring active:not-aria-[haspopup]:translate-y-px disabled:pointer-events-none disabled:bg-accent disabled:text-muted-foreground disabled:cursor-not-allowed aria-invalid:border-destructive aria-invalid:ring-2 aria-invalid:ring-destructive/20 dark:aria-invalid:border-destructive/50 dark:aria-invalid:ring-destructive/40 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4"`

Key changes: `rounded-lg` → `rounded-[1.875rem]`, `transition-all` → `transition-colors duration-200`, `ring-3` → `ring-2`, `disabled:opacity-50` → `disabled:bg-accent disabled:text-muted-foreground`.

- [ ] **Step 2: Update default variant**

**Old (line 12):** `"bg-primary text-primary-foreground [a]:hover:bg-primary/80"`

**New:** `"bg-primary text-primary-foreground hover:bg-secondary hover:text-secondary-foreground"`

- [ ] **Step 3: Update outline variant**

**Old (line 14):** `"border-border bg-background hover:bg-muted hover:text-foreground aria-expanded:bg-muted aria-expanded:text-foreground dark:border-input dark:bg-input/30 dark:hover:bg-input/50"`

**New:** `"border-border bg-background hover:bg-accent hover:text-foreground aria-expanded:bg-muted aria-expanded:text-foreground dark:border-input dark:bg-input/30 dark:hover:bg-input/50"`

- [ ] **Step 4: Update link variant**

**Old (line 21):** `"text-primary underline-offset-4 hover:underline"`

**New:** `"text-link underline-offset-4 hover:underline"`

- [ ] **Step 5: Update sizes**

Replace the entire `size` object (lines 23-35):

```typescript
size: {
  default:
    "h-10 gap-1.5 px-6 has-data-[icon=inline-end]:pr-2 has-data-[icon=inline-start]:pl-2",
  xs: "h-6 gap-1 rounded-[min(var(--radius-sm),10px)] px-2 text-xs in-data-[slot=button-group]:rounded-lg has-data-[icon=inline-end]:pr-1.5 has-data-[icon=inline-start]:pl-1.5 [&_svg:not([class*='size-'])]:size-3",
  sm: "h-8 gap-1 rounded-[min(var(--radius-md),12px)] px-4 text-[0.8rem] in-data-[slot=button-group]:rounded-lg has-data-[icon=inline-end]:pr-1.5 has-data-[icon=inline-start]:pl-1.5 [&_svg:not([class*='size-'])]:size-3.5",
  lg: "h-11 gap-1.5 px-8 has-data-[icon=inline-end]:pr-2 has-data-[icon=inline-start]:pl-2",
  icon: "size-10",
  "icon-xs":
    "size-6 rounded-[min(var(--radius-sm),10px)] in-data-[slot=button-group]:rounded-lg [&_svg:not([class*='size-'])]:size-3",
  "icon-sm":
    "size-8 rounded-[min(var(--radius-md),12px)] in-data-[slot=button-group]:rounded-lg",
  "icon-lg": "size-11",
},
```

Key changes: `h-8` → `h-10`, `h-9` → `h-11`, `h-7` → `h-8`, `px-2.5` → `px-6`/`px-4`/`px-8`.

- [ ] **Step 6: Commit**

```bash
git add src/frontend/src/components/ui/button.tsx
git commit -m "refactor(frontend): Nike pill buttons — rounded-full, flat, weight-500"
```

---

### Task 4: Restyle Card Component (Flat, 20px Radius)

**Files:**
- Modify: `src/frontend/src/components/ui/card.tsx`

- [ ] **Step 1: Update Card base (remove ring)**

**Old (line 15):** `"group/card flex flex-col gap-4 overflow-hidden rounded-xl bg-card py-4 text-sm text-card-foreground ring-1 ring-foreground/10 has-data-[slot=card-footer]:pb-0 has-[>img:first-child]:pt-0 data-[size=sm]:gap-3 data-[size=sm]:py-3 data-[size=sm]:has-data-[slot=card-footer]:pb-0 *:[img:first-child]:rounded-t-xl *:[img:last-child]:rounded-b-xl"`

**New:** `"group/card flex flex-col gap-4 overflow-hidden rounded-[1.25rem] bg-card py-4 text-sm text-card-foreground has-data-[slot=card-footer]:pb-0 has-[>img:first-child]:pt-0 data-[size=sm]:gap-3 data-[size=sm]:py-3 data-[size=sm]:has-data-[slot=card-footer]:pb-0 *:[img:first-child]:rounded-t-[1.25rem] *:[img:last-child]:rounded-b-[1.25rem]"`

Key changes: removed `ring-1 ring-foreground/10`, `rounded-xl` → `rounded-[1.25rem]`.

- [ ] **Step 2: Update CardHeader**

**Old (line 28):** `"group/card-header @container/card-header grid auto-rows-min items-start gap-1 rounded-t-xl px-4 group-data-[size=sm]/card:px-3 has-data-[slot=card-action]:grid-cols-[1fr_auto] has-data-[slot=card-description]:grid-rows-[auto_auto] [.border-b]:pb-4 group-data-[size=sm]/card:[.border-b]:pb-3"`

**New:** Same but `rounded-t-xl` → `rounded-t-[1.25rem]`.

- [ ] **Step 3: Update CardFooter**

**Old (line 84):** `"flex items-center rounded-b-xl border-t bg-muted/50 p-4 group-data-[size=sm]/card:p-3"`

**New:** `"flex items-center rounded-b-[1.25rem] border-t p-4 group-data-[size=sm]/card:p-3"`

Key change: removed `bg-muted/50` (Nike flat footer).

- [ ] **Step 4: Commit**

```bash
git add src/frontend/src/components/ui/card.tsx
git commit -m "refactor(frontend): Nike flat cards — no shadow/ring, 20px radius"
```

---

### Task 5: Restyle Remaining UI Components

**Files:**
- Modify: `src/frontend/src/components/ui/select.tsx`
- Modify: `src/frontend/src/components/ui/skeleton.tsx`
- Modify: `src/frontend/src/components/ui/progress.tsx`
- Modify: `src/frontend/src/components/ui/sonner.tsx`
- Modify: `src/frontend/src/components/ui/accordion.tsx`

- [ ] **Step 1: Select — change radius, remove shadow**

In `SelectTrigger` (line 39), change `rounded-lg` → `rounded-[0.5rem]` and `focus-visible:ring-3` → `focus-visible:ring-2`.

In `SelectContent` (line 65), change `rounded-lg` → `rounded-[0.5rem]` and remove `shadow-md`. Change `ring-1 ring-foreground/10` → `ring-1 ring-border`.

- [ ] **Step 2: Skeleton — use bg-secondary**

In `skeleton.tsx` (line 7), change `bg-muted` → `bg-secondary`.

- [ ] **Step 3: Progress — use bg-secondary for track**

In `progress.tsx` (line 15), change `bg-muted` → `bg-secondary`.

- [ ] **Step 4: Sonner — 20px radius**

In `sonner.tsx` (line 32), change `"--border-radius": "var(--radius)"` → `"--border-radius": "1.25rem"`.

- [ ] **Step 5: Accordion — ring-2**

In `accordion.tsx` (line 41), change `focus-visible:ring-3` → `focus-visible:ring-2` and `focus-visible:ring-ring/50` → `focus-visible:ring-ring`.

- [ ] **Step 6: Commit**

```bash
git add src/frontend/src/components/ui/
git commit -m "refactor(frontend): Nike styling for select, skeleton, progress, sonner, accordion"
```

---

### Task 6: Restyle FormField + AppNav + Layout

**Files:**
- Modify: `src/frontend/src/components/form-field.tsx`
- Modify: `src/frontend/src/components/app-nav.tsx`
- Modify: `src/frontend/src/app/layout.tsx`

- [ ] **Step 1: FormField — Nike input styling**

In `form-field.tsx`, replace the `inputClasses` constant (line 3):

**Old:** `"w-full rounded-md border border-input bg-background px-3 py-2 text-sm disabled:opacity-50"`

**New:** `"w-full rounded-[0.5rem] border border-input bg-secondary px-3 py-2 text-sm transition-colors duration-200 placeholder:text-muted-foreground focus-visible:border-foreground disabled:opacity-50"`

Also update the label class (appears 3 times): `text-sm font-medium` → `text-sm font-medium text-foreground`.

- [ ] **Step 2: AppNav — text-only nav (no bg highlight)**

In `app-nav.tsx`, replace all nav link `className` patterns.

For the nav items map (line 35), replace:

**Old:** `className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm transition-colors hover:bg-muted ${isActive ? "bg-muted font-medium" : "text-muted-foreground"}`}`

**New:** `className={`flex items-center gap-1.5 px-3 py-1.5 text-sm transition-colors hover:text-foreground ${isActive ? "text-foreground font-medium" : "text-muted-foreground"}`}`

For the profile link (line 47), same change: remove `rounded-md`, `hover:bg-muted` → `hover:text-foreground`, active: `bg-muted font-medium` → `text-foreground font-medium`.

For the logout button (line 55), same pattern: remove `rounded-md`, `hover:bg-muted` → `hover:text-foreground`.

For the login link (line 63), same pattern.

- [ ] **Step 3: Layout — solid header, wider max-width, taller nav**

In `layout.tsx` (line 29), replace the header:

**Old:** `<header className="sticky top-0 z-50 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">`

**New:** `<header className="sticky top-0 z-50 border-b border-border bg-background">`

In the header inner div (line 30):

**Old:** `<div className="mx-auto flex h-12 max-w-6xl items-center justify-between px-4">`

**New:** `<div className="mx-auto flex h-[60px] max-w-7xl items-center justify-between px-4">`

In the main element (line 38):

**Old:** `<main className="mx-auto w-full max-w-6xl p-4 sm:p-6">`

**New:** `<main className="mx-auto w-full max-w-7xl px-4 py-6 sm:px-6 sm:py-8">`

In the footer (line 39):

**Old:** `<footer className="border-t border-border px-4 py-3 text-center text-xs text-muted-foreground">`

**New:** `<footer className="px-4 py-6 text-center text-xs text-muted-foreground">`

Key changes: removed `border-t` from footer (Nike minimal footer), increased padding.

- [ ] **Step 4: Commit**

```bash
git add src/frontend/src/components/form-field.tsx src/frontend/src/components/app-nav.tsx src/frontend/src/app/layout.tsx
git commit -m "refactor(frontend): Nike nav (text-only), form inputs (bg-secondary), wider layout"
```

---

### Task 7: Restyle Pages — Typography + Spacing

**Files:**
- Modify: `src/frontend/src/app/page.tsx`
- Modify: `src/frontend/src/app/analyze/page.tsx`
- Modify: `src/frontend/src/components/dashboard/stats-cards.tsx`
- Modify: `src/frontend/src/components/dashboard/video-player.tsx`
- Modify: `src/frontend/src/components/placeholder-page.tsx`

- [ ] **Step 1: Home page — upload zone + settings cards**

In `page.tsx`:

Upload zone (line 190): change `rounded-lg` → `rounded-[1.25rem]` and `hover:border-primary` → `hover:border-foreground`.

Skater selection buttons (line 302): change `rounded` → `rounded-[0.5rem]`.

- [ ] **Step 2: Analyze page — processing heading**

In `analyze/page.tsx` (line 131): change `text-lg font-medium` → `nike-h2`.

- [ ] **Step 3: StatsCards — tighter grid, weight-500**

In `stats-cards.tsx` (line 41): change `gap-4` → `gap-3`.

In `stats-cards.tsx` (line 53): change `text-2xl font-bold` → `text-2xl font-medium`.

- [ ] **Step 4: VideoPlayer — 0px radius on video**

In `video-player.tsx` (line 12): change `rounded-md` → `rounded-none` on the `<video>` element.

- [ ] **Step 5: PlaceholderPage — Nike heading**

In `placeholder-page.tsx` (line 7): change `text-xl font-semibold` → `nike-h2`.

- [ ] **Step 6: Commit**

```bash
git add src/frontend/src/app/page.tsx src/frontend/src/app/analyze/page.tsx src/frontend/src/components/dashboard/ src/frontend/src/components/placeholder-page.tsx
git commit -m "refactor(frontend): Nike typography and spacing on all pages"
```

---

### Task 8: Restyle Auth + Profile Pages

**Files:**
- Modify: `src/frontend/src/app/(auth)/layout.tsx`
- Modify: `src/frontend/src/app/(auth)/login/page.tsx`
- Modify: `src/frontend/src/app/(auth)/register/page.tsx`
- Modify: `src/frontend/src/app/profile/page.tsx`
- Modify: `src/frontend/src/app/profile/settings/page.tsx`

- [ ] **Step 1: Auth layout — wider**

In `(auth)/layout.tsx` (line 4): change `max-w-sm` → `max-w-md`.

- [ ] **Step 2: Login page — Nike heading + link color**

In `login/page.tsx` (line 37): change `text-2xl font-bold` → `nike-h1`.

In `login/page.tsx` (line 65): change `text-primary hover:underline` → `text-link hover:underline`.

- [ ] **Step 3: Register page — Nike heading + link color**

In `register/page.tsx` (line 38): change `text-2xl font-bold` → `nike-h1`.

In `register/page.tsx` (line 75): change `text-primary hover:underline` → `text-link hover:underline`.

- [ ] **Step 4: Profile page — Nike heading**

In `profile/page.tsx` (line 56): change `text-2xl font-bold` → `nike-h1`.

- [ ] **Step 5: Settings page — Nike heading + theme buttons**

In `profile/settings/page.tsx` (line 65): change `text-2xl font-bold` → `nike-h1`.

Theme toggle buttons (line 98): change `rounded-md border` → `rounded-[0.5rem] border-[1.5px]`. Change active state `border-primary bg-primary/10` → `border-foreground bg-secondary`.

- [ ] **Step 6: Commit**

```bash
git add src/frontend/src/app/\(auth\)/ src/frontend/src/app/profile/
git commit -m "refactor(frontend): Nike typography on auth and profile pages"
```

---

### Task 9: Build Verification + Visual QA

- [ ] **Step 1: Run build**

```bash
cd src/frontend && bun run build
```

Expected: Build succeeds with no type errors.

- [ ] **Step 2: Run dev server and check pages**

```bash
cd src/frontend && bun run dev
```

Manual QA checklist:
- [ ] Light mode: home page upload zone, analyze page, profile, settings, login
- [ ] Dark mode: same pages
- [ ] Buttons are pill-shaped (30px radius)
- [ ] Cards have no shadow/ring, 20px radius
- [ ] Form inputs have gray background (#F5F5F5)
- [ ] Nav links use text-only active state (no bg highlight)
- [ ] Footer has no top border
- [ ] Links use blue (#1151FF / #1190FF in dark)
- [ ] Video player has 0px border-radius on video element
- [ ] Inter font renders (check Cyrillic on Russian locale)
- [ ] No shadows anywhere
- [ ] `font-bold` replaced with `font-medium` throughout

- [ ] **Step 3: Fix any issues found during QA**

- [ ] **Step 4: Final commit if fixes needed**

```bash
git add -A && git commit -m "fix(frontend): Nike design QA fixes"
```
