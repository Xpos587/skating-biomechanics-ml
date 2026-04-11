# Repo Cleanup & Best Practices Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring skating-biomechanics-ml repo tooling up to Control_Plane standards — git hooks, type checker, CI, git hygiene, and contributor docs.

**Architecture:** Incremental config-only changes. No source code modifications. Each task produces a self-contained commit. Order matters where later tasks depend on earlier ones (e.g., pyproject changes before CI updates).

**Tech Stack:** lefthook, basedpyright, ruff, gitleaks, GitHub Actions

---

## File Structure

| Action | File | Purpose |
|--------|------|---------|
| Create | `.gitmessage` | Commit message template with `type(scope): summary` format |
| Create | `.gitattributes` | Line ending enforcement for shell scripts |
| Modify | `data/models/.gitignore` | Add `*.onnx` to untrack model binaries |
| Modify | `.gitignore` | Add `*.onnx` global rule, cleanup |
| Modify | `lefthook.yml` | Add gitleaks, basedpyright, test group, .gitmessage integration |
| Modify | `pyproject.toml` | Replace mypy with basedpyright, improve ruff config |
| Modify | `.github/workflows/ci.yml` | Fix `mypy \|\| true`, use basedpyright, add gitleaks job |
| Create | `CONTRIBUTING.md` | Contributor guide adapted for this project |

---

### Task 1: Untrack ONNX models from git

**Files:**
- Modify: `data/models/.gitignore`
- Modify: `.gitignore`

The two ONNX files (17.9MB) are tracked in git but should be re-exported on demand via `scripts/export_models_to_onnx.py`.

- [ ] **Step 1: Update data/models/.gitignore**

Add `*.onnx` to the existing file:

```gitignore
# ONNX model weights (large binary files, re-export with scripts/export_models_to_onnx.py)
*.onnx
*.onnx.data

# PyTorch checkpoints (large binary files)
*.pth.tr
*.pt
```

- [ ] **Step 2: Untrack ONNX files from git index (keep on disk)**

Run:
```bash
git rm --cached data/models/TCPFormer_ap3d_81.onnx data/models/motionagformer-s-ap3d.onnx
```

- [ ] **Step 3: Verify files still exist on disk**

Run:
```bash
ls -la data/models/*.onnx
```
Expected: files still present (only untracked from git)

- [ ] **Step 4: Commit**

```bash
git add data/models/.gitignore
git commit -m "chore(repo): untrack ONNX models from git, re-export on demand"
```

---

### Task 2: Add .gitmessage commit template

**Files:**
- Create: `.gitmessage`

From Control_Plane — provides commit message format hint when running `git commit` without `-m`.

- [ ] **Step 1: Create .gitmessage**

```
# <type>(<scope>): <summary>
# types: feat|fix|refactor|chore|docs|test|ci
# scopes: pose|viz|tracking|analysis|pipeline|cli|models|repo
#
# Body (optional): what/why, links, migration notes.
```

- [ ] **Step 2: Set git template**

Run:
```bash
git config commit.template .gitmessage
```

- [ ] **Step 3: Verify**

Run:
```bash
git config commit.template
```
Expected: `.gitmessage`

- [ ] **Step 4: Commit**

```bash
git add .gitmessage
git commit -m "chore(repo): add commit message template"
```

---

### Task 3: Add .gitattributes

**Files:**
- Create: `.gitattributes`

From Control_Plane — enforces LF line endings for shell scripts.

- [ ] **Step 1: Create .gitattributes**

```
# Shell scripts must use LF line endings
scripts/*.sh text eol=lf
*.sh text eol=lf
```

- [ ] **Step 2: Commit**

```bash
git add .gitattributes
git commit -m "chore(repo): enforce LF line endings for shell scripts"
```

---

### Task 4: Switch mypy to basedpyright

**Files:**
- Modify: `pyproject.toml` (lines 99-123: replace `[tool.mypy]` with `[tool.basedpyright]`, update dev deps)

- [ ] **Step 1: Update pyproject.toml dev dependencies**

Replace `mypy>=1.0.0` with `basedpyright>=1.37.0` in `[dependency-groups] dev`:

```toml
[dependency-groups]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=6.0.0",
    "ruff>=0.8.0",
    "basedpyright>=1.37.0",
    "vulture>=2.0",
    "jupyter>=1.0.0",
    "onnxscript>=0.6.2",
]
```

- [ ] **Step 2: Replace [tool.mypy] section with [tool.basedpyright]**

Remove the entire `[tool.mypy]` and `[[tool.mypy.overrides]]` sections (lines 99-123). Replace with:

```toml
[tool.basedpyright]
typeCheckingMode = "standard"
reportMissingTypeStubs = false
reportUnknownMemberType = "warning"
reportUnknownVariableType = "warning"
reportUnknownParameterType = "warning"
reportUnknownArgumentType = "warning"
reportExplicitAny = "warning"
reportAny = "warning"
reportUnusedCallResult = "warning"
reportUnusedVariable = "warning"
pythonVersion = "3.11"
exclude = [
    "Sports2D/",
    "TCPFormer/",
]
```

- [ ] **Step 3: Improve ruff config — add [tool.ruff.format] section**

After the existing `[tool.ruff.lint.per-file-ignores]` block, add:

```toml
[tool.ruff.format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
line-ending = "auto"
```

- [ ] **Step 4: Improve ruff config — relax pylint limits**

Add after `[tool.ruff.lint]` ignore block (adapted from Control_Plane):

```toml
[tool.ruff.lint.pylint]
max-args = 12
max-branches = 35
max-returns = 6
max-statements = 80
```

This replaces the blanket `PLR0913`, `PLR0912`, `PLR0915`, `PLR0911` ignores. Remove those 4 lines from the `ignore` list:

```toml
ignore = [
    "E501",   # line too long (handled by formatter)
    "PLR2004", # magic value comparison
    "PLC0415", # import not at top-level (lazy imports for heavy deps)
    "SIM108", # ternary instead of if-else
    "ARG002", # unused method argument (common in callback interfaces)
    "RUF001", # ambiguous Cyrillic chars in strings (Russian UI text)
    "RUF002", # ambiguous Cyrillic chars in docstrings (Russian docs)
    "RUF003", # ambiguous Cyrillic chars in comments (Russian comments)
]
```

- [ ] **Step 5: Install new dependencies**

Run:
```bash
uv sync
```

- [ ] **Step 6: Verify basedpyright works**

Run:
```bash
uv run basedpyright --level error src/types.py
```
Expected: runs without crash (may have warnings, that's fine)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore(repo): switch mypy to basedpyright, improve ruff config"
```

---

### Task 5: Upgrade lefthook.yml

**Files:**
- Modify: `lefthook.yml`

Add gitleaks, basedpyright typecheck in hooks, and `test` group for manual execution (from Control_Plane).

- [ ] **Step 1: Replace lefthook.yml with upgraded version**

```yaml
# $schema: https://raw.githubusercontent.com/evilmartians/lefthook/master/schema.json
# Lefthook configuration for skating-biomechanics-ml
assert_lefthook_installed: true
colors: true

pre-commit:
  parallel: true
  commands:
    protect-branches:
      run: |
        BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null)
        if [[ $BRANCH =~ ^(main|master|develop)$ ]]; then
          echo "❌ Direct commits to '$BRANCH' are not allowed."
          echo "Please use a feature/hotfix/release branch."
          exit 1
        fi

    gitleaks:
      run: |
        if ! command -v gitleaks &> /dev/null; then
          exit 0
        fi
        if [ -f ".gitleaks-skip" ]; then
          exit 0
        fi
        gitleaks protect --staged --redact --no-banner

    ruff-check:
      glob: "*.py"
      run: uv run ruff check {staged_files} --fix
      stage_fixed: true

    ruff-format:
      glob: "*.py"
      run: uv run ruff format {staged_files}
      stage_fixed: true

commit-msg:
  commands:
    lint-commit-msg:
      run: |
        COMMIT_MSG=$(cat {1})

        if echo "$COMMIT_MSG" | grep -qE "^Merge (branch|remote-tracking branch)"; then
          exit 0
        fi

        PATTERN='^(feat|fix|refactor|chore|docs|test|ci)\(([a-z0-9_-]+)\): .{3,}$'

        if ! echo "$COMMIT_MSG" | grep -Eq "$PATTERN"; then
          echo "❌ Bad commit message:"
          echo "   $COMMIT_MSG"
          echo ""
          echo "✅ Use: type(scope): summary"
          echo "   Types: feat, fix, refactor, chore, docs, test, ci"
          echo "   e.g. feat(pose): add MotionAGFormer integration"
          echo "        fix(aligner): correct DTW window calculation"
          exit 1
        fi

pre-push:
  parallel: false
  commands:
    warn-pr-size:
      run: |
        REMOTE_BRANCH=$(git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null | sed 's|/.*||')
        REMOTE_BRANCH=${REMOTE_BRANCH:-origin}
        DEFAULT_BRANCH=$(git remote show $REMOTE_BRANCH 2>/dev/null | grep 'HEAD branch' | cut -d' ' -f5)
        DEFAULT_BRANCH=${DEFAULT_BRANCH:-master}

        CHANGED_FILES=$(git diff --stat $DEFAULT_BRANCH...HEAD 2>/dev/null | tail -n1 | awk '{print $1}')
        ADDED_LINES=$(git diff $DEFAULT_BRANCH...HEAD --numstat 2>/dev/null | awk '{add+=$1} END {print add}')
        DELETED_LINES=$(git diff $DEFAULT_BRANCH...HEAD --numstat 2>/dev/null | awk '{del+=$2} END {print del}')
        TOTAL_LINES=$((ADDED_LINES + DELETED_LINES))

        if [ -z "$TOTAL_LINES" ] || [ "$TOTAL_LINES" -eq 0 ]; then
          exit 0
        fi

        if [ "$TOTAL_LINES" -lt 100 ]; then
          SIZE="XS"
          EMOJI="🟢"
        elif [ "$TOTAL_LINES" -lt 400 ]; then
          SIZE="S"
          EMOJI="🟢"
        elif [ "$TOTAL_LINES" -lt 1000 ]; then
          SIZE="M"
          EMOJI="🟡"
        elif [ "$TOTAL_LINES" -lt 2000 ]; then
          SIZE="L"
          EMOJI="🟠"
          echo ""
          echo "⚠️  $EMOJI Large PR detected: ~$TOTAL_LINES lines changed in $CHANGED_FILES files"
          echo "   Size category: $SIZE"
          echo ""
          echo "   Consider splitting if this addresses multiple issues."
          echo ""
        else
          SIZE="XL"
          EMOJI="🔴"
          echo ""
          echo "🚨 $EMOJI Extra-large PR: ~$TOTAL_LINES lines changed in $CHANGED_FILES files"
          echo "   Size category: $SIZE (recommended max: 2000 lines)"
          echo ""
          echo "   Please consider splitting into smaller, focused PRs."
          echo ""
          echo "   To bypass: git push --no-verify"
          echo ""
        fi

        echo "📊 PR Size: $EMOJI $SIZE (~$TOTAL_LINES lines, $CHANGED_FILES files)"

# Manual groups
format:
  commands:
    ruff-format:
      run: uv run ruff check src/ tests/ scripts/ --fix && uv run ruff format src/ tests/ scripts/

test:
  commands:
    pytest:
      run: uv run pytest tests/ -v -m "not slow" --tb=short

typecheck:
  commands:
    basedpyright:
      run: uv run basedpyright --level error src/
```

- [ ] **Step 2: Verify lefthook hooks are installed**

Run:
```bash
lefthook install
```

- [ ] **Step 3: Test pre-commit hook**

Run:
```bash
echo "# test" >> .editorconfig && git add .editorconfig && git reset HEAD .editorconfig && git checkout -- .editorconfig
```
(Just verify lefthook runs, not a real commit)

- [ ] **Step 4: Commit**

```bash
git add lefthook.yml
git commit -m "chore(repo): upgrade lefthook — add gitleaks, test/typecheck groups"
```

---

### Task 6: Fix CI workflow

**Files:**
- Modify: `.github/workflows/ci.yml`

Fix: replace mypy with basedpyright, remove `|| true` bypass, add gitleaks job.

- [ ] **Step 1: Replace ci.yml**

```yaml
name: CI

on:
  pull_request:
    branches: [master]
  push:
    branches: [master]
  workflow_dispatch:

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

env:
  PYTHON_VERSION: "3.11"

jobs:
  lint:
    name: Lint (ruff)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5

      - name: Install uv
        uses: astral-sh/setup-uv@v6
        with:
          enable-cache: true

      - name: Setup Python
        uses: actions/setup-python@v6
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: uv sync --frozen --only-dev

      - name: Check formatting
        run: uv run ruff format --check src/ tests/ scripts/

      - name: Lint
        run: uv run ruff check src/ tests/ scripts/ --output-format=github

  typecheck:
    name: Type Check (basedpyright)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5

      - name: Install uv
        uses: astral-sh/setup-uv@v6
        with:
          enable-cache: true

      - name: Setup Python
        uses: actions/setup-python@v6
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: uv sync --frozen

      - name: Type check
        run: uv run basedpyright --level error src/

  gitleaks:
    name: Secret Scan (gitleaks)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
        with:
          fetch-depth: 0

      - name: gitleaks
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

  test:
    name: Tests
    runs-on: ubuntu-latest
    needs: [lint]
    steps:
      - uses: actions/checkout@v5

      - name: Install uv
        uses: astral-sh/setup-uv@v6
        with:
          enable-cache: true

      - name: Setup Python
        uses: actions/setup-python@v6
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: uv sync --frozen

      - name: Run tests
        run: uv run pytest tests/ -v -m "not slow" --tb=short

  ci-passed:
    name: CI Passed
    runs-on: ubuntu-latest
    needs: [lint, typecheck, gitleaks, test]
    if: always()
    steps:
      - name: Check results
        run: |
          if [[ "${{ needs.lint.result }}" != "success" ]] || \
             [[ "${{ needs.typecheck.result }}" != "success" ]] || \
             [[ "${{ needs.gitleaks.result }}" != "success" ]] || \
             [[ "${{ needs.test.result }}" != "success" ]]; then
            echo "::error::CI failed"
            exit 1
          fi
          echo "✅ All CI checks passed!"
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci(repo): replace mypy with basedpyright, add gitleaks, fix typecheck gate"
```

---

### Task 7: Create CONTRIBUTING.md

**Files:**
- Create: `CONTRIBUTING.md`

Adapted from Control_Plane for this project's context.

- [ ] **Step 1: Create CONTRIBUTING.md**

```markdown
# Contributing

## TL;DR

1. Work **only via feature branches** (push to `master` is blocked by lefthook)
2. Follow commit format: `type(scope): summary`
3. Install hooks: `lefthook install`
4. Before PR: `lefthook run test` and `lefthook run typecheck`

---

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- [lefthook](https://github.com/evilmartians/lefthook) git hooks manager
- [gitleaks](https://github.com/gitleaks/gitleaks) secret scanner
- CUDA-capable GPU (for inference)

### Install

```bash
# Clone and enter project
git clone <repo-url> && cd skating-biomechanics-ml

# Install dependencies
uv sync

# Setup CUDA compatibility (required after uv sync)
bash scripts/setup_cuda_compat.sh

# Install git hooks
lefthook install

# Set commit message template
git config commit.template .gitmessage
```

### Verify

```bash
uv run pytest tests/ -v -m "not slow" --tb=short
uv run ruff check src/ tests/ scripts/
uv run basedpyright --level error src/
```

---

## Git Flow

### Branches

```bash
git checkout master
git pull
git checkout -b <username>/<feature-name>
```

### Commits

Format: `type(scope): summary`

| Type | Use for |
|------|---------|
| feat | New feature |
| fix | Bug fix |
| refactor | Code restructuring |
| chore | Tooling, deps, config |
| docs | Documentation |
| test | Tests |
| ci | CI/CD |

Scopes: `pose`, `viz`, `tracking`, `analysis`, `pipeline`, `cli`, `models`, `repo`

Examples:
```
feat(pose): add MotionAGFormer integration
fix(aligner): correct DTW window calculation
chore(repo): upgrade ruff to v0.14
```

### Git Hooks

Lefthook runs automatically:

- **pre-commit**: branch protection, gitleaks, ruff lint/format
- **commit-msg**: conventional commit format validation
- **pre-push**: PR size warning

Manual groups:
```bash
lefthook run format    # Format all Python files
lefthook run test      # Run test suite
lefthook run typecheck # Run basedpyright
```

---

## Code Quality

### Linting & Formatting

Ruff handles both. Config in `pyproject.toml`.

```bash
uv run ruff check src/ tests/ scripts/ --fix  # Lint
uv run ruff format src/ tests/ scripts/        # Format
```

### Type Checking

Basedpyright (strict fork of pyright). Config in `pyproject.toml`.

```bash
uv run basedpyright --level error src/
```

### Testing

Pytest with coverage. 431+ tests.

```bash
uv run pytest tests/ -v                        # All tests
uv run pytest tests/ -v -m "not slow"          # Skip slow tests
uv run pytest tests/ -v -k "test_tracker"      # By name
uv run pytest tests/ --cov=src --cov-report=html  # With coverage
```

### Key Conventions

- **HALPE26 (26kp)** for 2D pose, **H3.6M (17kp)** for 3D
- `poses_norm` = normalized [0,1], `poses_px` = pixel coordinates
- GPU-only inference: always `device='cuda'`
- Russian text for UI output and recommendations
- See `CLAUDE.md` for full architecture details

---

## PR Process

1. Update from master: `git fetch origin && git merge origin/master`
2. Push: `git push -u origin <branch>`
3. Create PR on GitHub
4. Wait for CI (lint → typecheck → gitleaks → test)
5. Get review
6. Merge
```

- [ ] **Step 2: Commit**

```bash
git add CONTRIBUTING.md
git commit -m "docs(repo): add CONTRIBUTING.md"
```

---

### Task 8: Clean up .gitignore

**Files:**
- Modify: `.gitignore`

Minor cleanup: add `*.onnx` global rule, add `vast-instances.json` pattern (currently present on disk), ensure consistency.

- [ ] **Step 1: Update .gitignore**

Add these entries to the appropriate sections:

```gitignore
# Models (global)
*.onnx

# Vast.ai
vast-instances.json
```

Also remove the duplicate `data/models/*.pth.tr` line (already covered by `*.pth` and `*.pt` patterns).

- [ ] **Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore(repo): cleanup .gitignore — add *.onnx, vast.ai"
```

---

## Verification

After all tasks are complete, verify the full setup:

```bash
# 1. Hooks work
lefthook install
lefthook run format
lefthook run test

# 2. Type checker works
uv run basedpyright --level error src/

# 3. ONNX models are untracked
git ls-files | grep '.onnx'
# Expected: no output

# 4. Commit template is set
git config commit.template
# Expected: .gitmessage

# 5. Git status is clean
git status
# Expected: clean working tree (all committed)

# 6. Full test suite
uv run pytest tests/ -v -m "not slow" --tb=short
```
