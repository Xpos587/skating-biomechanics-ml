# experiments/CLAUDE.md — ML Experiments

## Purpose

Exploratory ML experiments. Not part of the production pipeline (`ml/skating_ml/`).
These are scripts and notebooks for testing hypotheses before integrating into the main system.

## Directory Convention

```
experiments/
├── README.md              # Master report: hypotheses, results table, conclusions
├── CLAUDE.md              # This file
├── checkpoints/           # Model checkpoints (not in git)
├── exp_<short_name>.py    # Experiment scripts
└── YYYY-MM-DD-<topic>.md  # Standalone experiment reports
```

## Experiment Template

Every experiment script must document at the top:

```python
"""
Experiment: <short name>
Hypothesis: <what you're testing>
Status: PENDING | CONFIRMED | REJECTED | INCONCLUSIVE

Usage:
    uv run python experiments/exp_<name>.py
"""
```

Every experiment report must include:
1. **Hypothesis** — what you're testing and expected outcome
2. **Method** — model, data, config
3. **Results** — metrics table with numbers
4. **Conclusion** — confirmed/rejected/inconclusive with reasoning

## Checkpoints

Save to `experiments/checkpoints/<exp_name>/`:
- `best.pt` — best model (highest validation metric)
- `epoch_<N>.pt` — specific epoch snapshots
- `config.json` — full hyperparameters for reproducibility
- `training_log.csv` — per-epoch metrics

```python
import torch

CHECKPOINT_DIR = Path("experiments/checkpoints/exp_name")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "best_metric": best_acc,
    "config": vars(args),
}, CHECKPOINT_DIR / "best.pt")
```

`experiments/checkpoints/` is in `.gitignore` — do not commit model weights.

## Hypothesis Tracking

All hypotheses live in `README.md` master table:

| ID | Hypothesis | Status | Evidence |
|----|-----------|--------|----------|
| H0 | ... | PENDING/CONFIRMED/REJECTED | Link to experiment |

Status values: `PENDING` → `CONFIRMED` | `REJECTED` | `INCONCLUSIVE`

## Categories

When adding a new experiment, classify it under one of these categories:
- **classification** — element type classification
- **pose-estimation** — pose extraction quality
- **tracking** — multi-person tracking
- **biomechanics** — metrics, phase detection, physics
- **visualization** — rendering, overlays, comparison
- **data** — preprocessing, augmentation, datasets

## Running

```bash
uv run python experiments/exp_<name>.py
```

Requires `torch` with CUDA. Datasets must be in `data/datasets/`.
