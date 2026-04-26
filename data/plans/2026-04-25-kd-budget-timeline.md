# KD Project Budget & Timeline Analysis

**Date:** 2026-04-25
**Agent:** Budget & Timeline (Agent 5)
**Scope:** Complete budget audit, scenario planning, and timeline for MogaNet-B -> YOLO26-Pose KD project

---

## 1. Current Spend Audit

### What has been spent (v1-v35d)

| Phase | Description | Estimated Cost | Notes |
|-------|-------------|---------------|-------|
| v1-v34 iterations | Early experiments, data prep, heatmap generation | ~$15-20 | Multiple short runs, debugging |
| Pre-flight checks | FineFS quality, sigma head POC, HDF5 benchmark | ~$1-2 | Short GPU sessions |
| Teacher heatmap generation | 264K heatmaps, MogaNet-B inference | ~$3-4 | ~3h on RTX 5090 ($0.59/hr) |
| Teacher feature generation | Backbone features for DWPose Stage 1 | ~$2-3 | ~4h on RTX 5090 |
| v35c training | 7 epochs (ABORTED -- KD inactive, val/cls_loss spikes) | ~$1-2 | Wasted: bugs made training useless |
| Data upload | One-time upload to Vast.ai instances | $0 | Non-GPU time only |
| **Total spent** | | **~$22-31** | |

### Estimated remaining budget

| Scenario | Remaining | Notes |
|----------|-----------|-------|
| Total budget | $150 | |
| Minus worst-case spend | -$31 | Conservative estimate |
| **Remaining (conservative)** | **$119** | |
| **Remaining (optimistic)** | **$128** | If actual spend was lower |

### Wasted spend analysis

| Waste Item | Cost | Root Cause | Preventable? |
|-----------|------|-----------|-------------|
| v35c 7-epoch run | ~$1-2 | 8 CRITICAL bugs in distill_trainer.py | Partially: bugs should have been caught in unit tests |
| Sigmoid heatmap gen (v5) | ~$1 | Applied sigmoid to raw logits, destroyed peaks | Yes: should have validated with visual inspection before batch |
| Multiple short debug runs | ~$5 | Iterative debugging on GPU | Partially: some GPU debugging unavoidable |
| **Total waste** | **~$7-8** | | |
| **Waste ratio** | **~23-26%** | | |

---

## 2. Vast.ai Current Pricing (April 2026)

### Source: vast.ai/pricing, gpunex.com review, synpixcloud.com comparison

| GPU | VRAM | On-Demand (from) | Verified (from) | Disk (800GB) | Effective with disk |
|-----|------|-------------------|-----------------|--------------|-------------------|
| **RTX 4090** | 24GB | $0.30/hr | $0.34/hr | ~$0.22/hr | **$0.52/hr** |
| **RTX 5090** | 32GB | $0.36/hr | $0.40/hr | ~$0.22/hr | **$0.58/hr** |
| **A100 40GB** | 40GB | $0.52/hr | $0.60/hr | ~$0.22/hr | **$0.74/hr** |
| **H100 SXM** | 80GB | $0.90/hr | $1.60/hr | ~$0.22/hr | **$1.12-1.82/hr** |

### Key pricing insights

1. **Marketplace volatility:** Budget 20-40% above lowest advertised rate (GPUnex review). Unverified hosts effective cost is 20-40% higher due to downtime/restarts.
2. **Verified vs unverified:** Verified hosts cost ~15% more but deliver consistent uptime. For jobs > 4 hours, verified is usually better value.
3. **Storage adds significantly:** 800GB disk at ~$0.22/hr adds 40-60% to GPU-only price. The project needs ~200-300GB for datasets + heatmaps + features. 400GB disk would save ~$0.11/hr.
4. **Spot/interruptible:** 50%+ cheaper but can be killed mid-training. Only suitable for resumable workloads with frequent checkpointing.

### Recommended GPU for this project

**RTX 4090 verified, 400GB disk** -- estimated ~$0.55/hr effective.
- Sufficient VRAM for YOLO26s (batch=128, imgsz=384)
- 24GB handles teacher heatmap batch loading
- Best price/performance ratio for this workload size
- 400GB disk fits all datasets + heatmaps (~200GB total)

---

## 3. v35d Completion Cost

### Current v35d status

v35d is described as running with fixed bugs from v35c review. However, the v35c multi-agent review found **8 CRITICAL bugs** that make KD training completely non-functional. v35d must have fixed these to be meaningful.

**Assuming v35d has the critical fixes applied:**

| Parameter | Value |
|-----------|-------|
| Model | YOLO26-Pose Small |
| Epochs | 210 |
| Batch size | 128 |
| Image size | 384 |
| Speed | ~3.5 it/s (observed) |
| Time per epoch | ~10 min (observed) |
| GPU | RTX 4090 or RTX 5090 |

### v35d cost estimate

| Item | Time | Cost (RTX 4090 @ $0.55/hr) |
|------|------|---------------------------|
| Training (210 epochs) | 35h | $19.25 |
| Validation overhead | ~5h | $2.75 |
| Checkpointing | ~0.5h | $0.28 |
| Data loading variance | ~5h | $2.75 |
| **Total v35d** | **~45.5h** | **~$25.00** |

### GPU comparison for v35d

| GPU | Speedup vs 4090 | Effective $/hr | v35d Cost | Verdict |
|-----|-----------------|----------------|-----------|---------|
| RTX 4090 | 1.0x | $0.55 | $25.00 | **Recommended** -- best value |
| RTX 5090 | ~1.3x | $0.58 | $21.50 | Marginal improvement, 8% more expensive per hour |
| A100 40GB | ~1.5x | $0.74 | $23.00 | Faster but significantly more expensive per hour |
| H100 SXM | ~2.5x | $1.50 | $18.75 | Fastest but poor price/performance for this workload |

**Recommendation:** Stick with RTX 4090. The workload is I/O-bound (HDF5/LMDB reads), not compute-bound. Faster GPUs show diminishing returns.

---

## 4. Scenario-Based Budget Allocation

### Scenario A: v35d Succeeds (pose mAP50 > 0.6 at epoch 100)

**Probability: 30%** (optimistic -- requires all 8 critical bugs to be properly fixed AND KD to produce meaningful signal)

| Task | Cost | Notes |
|------|------|-------|
| v35d completion | $25.00 | Already budgeted above |
| ONNX export + validation | $1-2 | 1-2h GPU for export + test inference |
| TensorRT optimization | $2-3 | 2-3h GPU for TRT build + benchmark |
| Production A/B testing | $3-5 | Compare vs RTMO baseline on 10 test videos |
| Integration testing | $2-3 | End-to-end pipeline test with tracking |
| **Total Scenario A** | **$33-38** | |

**Cumulative spend:** $31 (past) + $38 (scenario A) = **$69** (46% of budget)
**Remaining after A:** $81

### Scenario B: v35d Mediocre (pose mAP50 0.3-0.6)

**Probability: 40%** (most likely -- bugs partially fixed, KD contributes some signal but below expectations)

| Task | Cost | Notes |
|------|------|-------|
| v35d completion (full 210 ep) | $25.00 | May need to let it run full course |
| Debugging + analysis | $2-3 | Inspect loss curves, per-keypoint AP |
| Direct fine-tuning baseline | $10-15 | Same data, no KD -- isolate KD contribution |
| Hyperparameter tuning | $10-15 | KD weight sweep, LR schedule, alpha/beta |
| Data augmentation experiments | $5-8 | Ice-specific augmentations |
| Re-run with best config | $15-20 | Second full training run |
| **Total Scenario B** | **$67-86** | |

**Cumulative spend:** $31 (past) + $86 (scenario B) = **$117** (78% of budget)
**Remaining after B:** $33 (tight but sufficient for Scenario B completion)

### Scenario C: v35d Fails (pose mAP50 < 0.3)

**Probability: 20%** (critical bugs remain or data quality issue)

| Task | Cost | Notes |
|------|------|-------|
| v35d (abort at epoch 30) | $4-5 | Early abort saves ~$20 |
| Root cause analysis | $1-2 | Local debugging (CPU), minimal GPU |
| Direct fine-tuning (no KD) | $10-15 | Fallback: simple fine-tuning on skating data |
| RTMO benchmark comparison | $2-3 | Is the problem the approach or the baseline? |
| Alternative: RTMPose fine-tuning | $10-15 | Fine-tune RTMPose directly on skating data |
| Pseudo-labeling prep | $5-10 | Generate labels for SkatingVerse 28K videos |
| Re-train with expanded data | $15-20 | More data may be the answer |
| **Total Scenario C** | **$47-70** | |

**Cumulative spend:** $31 (past) + $70 (scenario C) = **$101** (67% of budget)
**Remaining after C:** $49

### Scenario D: v35d Shows Promise but Needs More Data

**Probability: 10%** (KD works but plateau due to limited training data)

| Task | Cost | Notes |
|------|------|-------|
| v35d completion | $25.00 | Get the best model from current data |
| Pseudo-labeling SkatingVerse | $20-30 | Run MogaNet-B on 28K competition videos |
| Data filtering + QC | $2-3 | Remove low-confidence pseudo-labels |
| Re-train with expanded data | $25-30 | Full KD training with 2-3x more data |
| **Total Scenario D** | **$72-88** | |

**Cumulative spend:** $31 (past) + $88 (scenario D) = **$119** (79% of budget)
**Remaining after D:** $31

### Budget Summary Table

| Scenario | Probability | Additional Cost | Total Project Cost | Budget OK? |
|----------|-------------|----------------|-------------------|------------|
| A: v35d succeeds | 30% | $33-38 | $64-69 | Yes (46%) |
| B: v35d mediocre | 40% | $67-86 | $98-117 | Yes (78%) |
| C: v35d fails | 20% | $47-70 | $78-101 | Yes (67%) |
| D: Needs more data | 10% | $72-88 | $103-119 | Tight (79%) |

---

## 5. Parallel Tracks Budget

While v35d trains (~35h GPU = ~$25), several tasks can run in parallel on separate instances or locally.

### What can run in parallel

| Track | GPU Required? | Cost | Wall Time | Dependency |
|-------|---------------|------|-----------|------------|
| **v35d training** | Yes (primary) | $25 | 1.5 days | Running now |
| **Direct fine-tuning baseline** | Yes (2nd instance) | $12-15 | 1.5 days | Independent |
| **RTMO benchmark** | Yes (can share instance) | $2-3 | 4h | Independent |
| **Data preprocessing** | No (CPU) | $0 | 2-4h | Local machine |
| **Pseudo-labeling prep** | Yes (3rd instance) | $5-10 | 0.5 days | Independent |
| **Bug fixes + unit tests** | No (CPU) | $0 | 1 day | Local |
| **ONNX export testing** | Yes (brief) | $1-2 | 1h | Needs trained model |

### Recommended parallel strategy

**Phase 1 (Days 1-2): Maximize parallelism**

```
Instance 1 (RTX 4090): v35d training ($25, 1.5 days)
Instance 2 (RTX 4090): Direct fine-tuning baseline ($12, 1.5 days)
Local machine: Bug fixes, unit tests, data analysis (free)
```

**Total Phase 1 cost:** $37 for both instances running in parallel.
**Savings vs sequential:** ~$12 (1.5 days saved on instance 2 wait time).

**Phase 2 (Day 2-3): Evaluation**

After v35d completes, use Instance 1 for:
- RTMO benchmark ($2-3, 4h)
- ONNX export testing ($1-2, 1h)
- Results analysis ($0, local)

**Phase 3 (Days 3-5): Conditional on results**

Depends on which scenario plays out (A/B/C/D from Section 4).

### Parallel budget summary

| Phase | Instances | Duration | Cost |
|-------|-----------|----------|------|
| Phase 1: Parallel training | 2x RTX 4090 | 1.5 days | $37 |
| Phase 2: Evaluation | 1x RTX 4090 | 0.5 days | $7 |
| Phase 3: Conditional | 1x RTX 4090 | 1-3 days | $15-45 |
| **Total (typical)** | | **3-5 days** | **$59-89** |

---

## 6. GPU Instance Strategy

### Instance selection criteria

| Criterion | Priority | Notes |
|-----------|----------|-------|
| Reliability | HIGH | Lost training = wasted budget |
| Price | HIGH | $150 total budget is hard constraint |
| VRAM | MEDIUM | 24GB sufficient for YOLO26s batch=128 |
| Disk | MEDIUM | Need ~200-300GB for data |
| Location | LOW | Latency doesn't matter for training |

### Recommended configuration

```
GPU: RTX 4090 (verified)
VRAM: 24GB
Disk: 400GB (not 800GB -- save $0.11/hr)
Image: CUDA 12 + PyTorch 2.x
Verification: Verified preferred (15% premium, worth it for >4h jobs)
Region: Any with >100 Mbps bandwidth
```

### Checkpoint safety protocol

1. **Save every 10 epochs** to local disk
2. **Sync best model to R2** after each improvement (free, non-GPU)
3. **Training script must be resumable** from any checkpoint
4. **Monitor instance health** via `nvidia-smi` every hour (script)
5. **Kill switch:** If GPU temp > 90C or clock drops >20%, destroy instance

### Spot instance policy

**Do NOT use spot/interruptible for this project.** Rationale:
- Training is 35+ hours continuous
- Spot instances have ~30% interruption rate for long jobs
- Checkpoint resume wastes ~30 min per interruption
- Net savings: ~50% on hourly rate - 30% wasted time = only ~20% effective savings
- Not worth the reliability risk for a $150 budget

---

## 7. Timeline (Calendar Time)

### Critical path

```
Day 0 (today):
  - v35d running (or about to start)
  - Fix remaining bugs if any

Day 1-2 (36h):
  - v35d training completes
  - Parallel: direct fine-tuning baseline on instance 2
  - Local: analyze v35c/v35d loss curves, identify issues

Day 2-3 (12h):
  - RTMO benchmark comparison
  - ONNX export + inference test
  - Results compilation

Day 3-4 (24h):
  - Decision point: which scenario?
  - If A: integration testing
  - If B: hyperparameter sweep
  - If C: pivot to direct fine-tuning or RTMPose
  - If D: pseudo-labeling SkatingVerse

Day 4-6 (48h):
  - Execute scenario-specific plan
  - Final evaluation
  - Production integration

Day 6-7 (24h):
  - Documentation
  - Final model selection
  - Deployment prep
```

### Timeline by scenario

| Scenario | Total Calendar Days | GPU Hours | Total Cost |
|----------|--------------------|-----------|------------|
| **A: v35d succeeds** | 3-4 days | 50-60h | $59-69 |
| **B: v35d mediocre** | 5-7 days | 100-150h | $89-117 |
| **C: v35d fails** | 5-7 days | 80-120h | $78-101 |
| **D: Needs more data** | 7-10 days | 130-170h | $103-119 |

### Gantt-style timeline

```
           Day 1   Day 2   Day 3   Day 4   Day 5   Day 6   Day 7
Instance 1 [v35d training 35h   ]
Instance 2 [finetune baseline    ]
Local     [bug fixes][analysis  ]
           ─────────────────────────────────────────────────────────
           Day 1   Day 2   Day 3   Day 4   Day 5   Day 6   Day 7
Inst. 1    ............[RTMO bench][eval    ]
Inst. 2    ............[ONNX test ]
Local     ............[results   ]
           ─────────────────────────────────────────────────────────
Scenario B:
Inst. 1    ....................[HP sweep    ][re-train     ]
Inst. 2    ....................[augment exp ]
```

---

## 8. Cost Optimization

### Cheapest path to production-grade pose

**Option 1: Skip KD, direct fine-tuning** -- $15-20 total
- Fine-tune YOLO26s on skating data (FineFS + AP3D-fs)
- No teacher model, no heatmaps, no complex pipeline
- Expected AP: 0.65-0.75 (vs 0.80+ with successful KD)
- Risk: Lower quality but 3-5x cheaper

**Option 2: Minimal KD** -- $30-40 total
- Fix critical bugs, run 50 epochs (not 210)
- Early stopping at patience=20
- If AP plateaus, stop immediately
- Expected AP: 0.60-0.70

**Option 3: Full KD pipeline** -- $60-90 total (current plan)
- Fix all bugs, run full 210 epochs
- Parallel baselines, hyperparameter tuning
- Expected AP: 0.75-0.85 (if KD works)

### What to cut if budget runs low

| Priority | Cut Item | Savings | Impact |
|----------|---------|---------|--------|
| 1 | Reduce epochs 210->50 | $18 | May not converge |
| 2 | Skip feature distillation | $5 | Coordinate-only KD is standard |
| 3 | Skip direct fine-tuning baseline | $12-15 | Lose KD contribution measurement |
| 4 | Skip pseudo-labeling | $20-30 | Limited data diversity |
| 5 | Skip TensorRT optimization | $2-3 | Slightly slower inference |
| 6 | Use unverified instances | 15-20% | Risk of instance failure |

### Free alternatives

| Task | GPU Cost | CPU Alternative |
|------|---------|----------------|
| Bug fixes + unit tests | $0 | Local machine (CPU) |
| Data analysis + visualization | $0 | Local machine (CPU) |
| Data preprocessing | $0 | Local machine (CPU) |
| Loss curve analysis | $0 | TensorBoard locally |
| Small-scale inference tests | $0 | ONNX Runtime CPU |
| Document writing | $0 | Local machine |

### Budget monitoring strategy

```python
# Checkpoint every 10 epochs, log cumulative cost
BUDGET_ALERT_THRESHOLDS = {
    0.50: "50% budget used -- check if on track",
    0.75: "75% budget used -- evaluate whether to continue",
    0.90: "90% budget used -- CRITICAL: stop or commit to finish",
}

def check_budget(spent, total=150):
    ratio = spent / total
    if ratio in BUDGET_ALERT_THRESHOLDS:
        print(f"ALERT: {BUDGET_ALERT_THRESHOLDS[ratio]} (${spent:.2f} / ${total})")
    if spent > total * 0.75:
        # Check if val AP justifies continued spending
        if current_ap < 0.4:
            print("ABORT: Budget >75% used and AP < 0.4")
            return False
    return True
```

---

## 9. Risk-Adjusted Budget

### Expected value calculation

| Scenario | Probability | Cost | Probability * Cost |
|----------|-------------|------|-------------------|
| A: Success | 30% | $69 | $20.70 |
| B: Mediocre | 40% | $117 | $46.80 |
| C: Failure | 20% | $101 | $20.20 |
| D: More data | 10% | $119 | $11.90 |
| **Expected total cost** | | | **$99.60** |

**Expected cost: ~$100** (67% of $150 budget)

### Budget overrun probability

| Budget Cap | Probability of Exceeding | Notes |
|-----------|-------------------------|-------|
| $80 | 30% | Scenario A fits easily |
| $100 | 50% | Expected value |
| $120 | 80% | Scenarios B, C, D may approach this |
| $150 | 5% | Only if all contingencies trigger |

### Recommended reserve

| Reserve Type | Amount | Purpose |
|-------------|--------|---------|
| Contingency (instance failure, re-runs) | $20 | 2-3 extra hours of GPU |
| Debug buffer (unexpected issues) | $15 | Additional debugging sessions |
| Opportunity reserve (pseudo-labeling, new ideas) | $16 | Optional improvements |
| **Total reserve** | **$51** | 34% of budget |

---

## 10. Final Recommended Budget Allocation

### Recommended spend plan

| Category | Allocation | % of Budget | Notes |
|----------|-----------|-------------|-------|
| v35d training (primary) | $25 | 17% | Core experiment |
| Direct fine-tuning baseline (parallel) | $12 | 8% | Isolate KD contribution |
| RTMO benchmark | $3 | 2% | Comparison baseline |
| Bug fixes + evaluation | $5 | 3% | Debugging sessions |
| Hyperparameter tuning (if needed) | $15 | 10% | Scenario B trigger |
| Pseudo-labeling (if needed) | $25 | 17% | Scenario D trigger |
| Production integration | $8 | 5% | ONNX, TensorRT, testing |
| **Contingency reserve** | **$32** | **21%** | Unexpected issues |
| **Opportunity reserve** | **$25** | **17%** | New ideas, extra experiments |
| **TOTAL** | **$150** | **100%** | |

### Decision gates

| Gate | Trigger | Action |
|------|---------|--------|
| **Gate 1** (Day 2) | v35d epoch 30: AP < 0.2 | STOP v35d. Investigate data quality. Consider Scenario C. |
| **Gate 2** (Day 3) | v35d epoch 100: AP < 0.4 | Evaluate. Consider reducing to 50 epochs max. |
| **Gate 3** (Day 3) | v35d epoch 100: AP 0.4-0.6 | Continue but plan for Scenario B. Start HP sweep. |
| **Gate 4** (Day 3) | v35d epoch 100: AP > 0.6 | Full speed ahead. Plan for Scenario A integration. |
| **Budget gate** (any time) | Spent > $75 and AP < 0.5 | ABORT. Budget insufficient for meaningful improvement. |

### Critical success factors

1. **Fix all 8 CRITICAL bugs before starting v35d** -- otherwise the entire $25 is wasted
2. **Run direct fine-tuning baseline in parallel** -- without it, we cannot measure KD contribution
3. **Set up checkpoint sync to R2** -- protects against instance failure
4. **Monitor GPU health** -- thermal throttling wastes budget silently
5. **Stop early if metrics don't improve** -- patience=30 epochs, budget gate at $75

---

## Appendix: Cost Tracking Template

```
Date       | Instance | GPU       | Hours | $/hr | Total $ | Task                    | AP
-----------|----------|-----------|-------|------|---------|-------------------------|----
2026-04-18 | vast_123 | RTX 5090  | 3.0   | 0.59 | $1.77   | Heatmap generation      | n/a
2026-04-19 | vast_123 | RTX 5090  | 4.0   | 0.59 | $2.36   | Feature generation      | n/a
2026-04-20 | vast_123 | RTX 5090  | 1.5   | 0.59 | $0.89   | v35c training (abort)   | 0.069
...
CUMULATIVE: $31.00 remaining: $119.00
```
