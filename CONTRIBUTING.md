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