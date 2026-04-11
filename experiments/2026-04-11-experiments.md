# Experiment Log — Original Notes (Superseded)

> **This file contains raw notes from the initial experiment session.**
> **The canonical report is at [README.md](README.md).**

All hypotheses, results, training curves, and conclusions have been consolidated into `README.md` with proper formatting, hyperparameter documentation, and cross-experiment analysis.

### Quick Reference

| Experiment | Result | Verdict |
|-----------|--------|---------|
| Exp 1 — Cosine similarity | gap 0.09 | H2 REJECTED |
| Exp 2 — 1D-CNN 64 cls | 21.8% | H1 REJECTED |
| Exp 2b — CNN top-10 | 70.4% | Class imbalance confirmed |
| Exp 2c — BiGRU 64 cls | 61.6% | Variable-length critical |
| Exp 2d — MMFS BiGRU | 65.4% | FSC preferred |
| Exp 2e — BiGRU top-10 | 81.5% | 80% target achievable |
| Exp 3 — Augmentation | 67.9% best | H4 REJECTED |
