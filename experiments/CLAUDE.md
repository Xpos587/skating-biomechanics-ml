# experiments/CLAUDE.md — ML Experiments

## Purpose

Jupyter-style experiments for figure skating element classification. These are exploratory scripts, not part of the production pipeline.

## Files

| File | Purpose |
|------|---------|
| `README.md` | Experiment overview and goals |
| `exp_2b_2c_2d.py` | PyTorch classification: BiGRU temporal model, quality scoring |
| `exp_augmentation.py` | Data augmentation experiments |

## Target

80% classification accuracy on figure skating elements, running on RTX 3050 Ti GPU.

## Running

```bash
uv run python experiments/exp_2b_2c_2d.py
```
