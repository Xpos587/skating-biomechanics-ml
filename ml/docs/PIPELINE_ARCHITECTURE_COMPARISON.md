# Pipeline Architecture Comparison

## Current vs. Proposed Parallelization

### CURRENT ARCHITECTURE (Sequential)

```
┌─────────────────────────────────────────────────────────────────┐
│                    SEQUENTIAL PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Video ──▶ RTMO ──▶ Gap ──▶ Norm ──▶ Smooth ──▶ 3D Lift         │
│  (5.6s)   (5.6s) (0.8s) (0.3s)   (0.5s)    (1.5s)              │
│                                                                   │
│                └─▶ Phase Detect (0.8s)                          │
│                     └─▶ Metrics (1.2s)                          │
│                           └─▶ Ref Load (I/O)                    │
│                                └─▶ DTW (0.9s)                   │
│                                   └─▶ Recs (0.4s)              │
│                                                                   │
│  TOTAL TIME: 12.0s                                              │
│  PARALLEL STAGES: 2 (3D lift, phase detect)                    │
└─────────────────────────────────────────────────────────────────┘
```

### PROPOSED ARCHITECTURE (Enhanced Parallel)

```
┌─────────────────────────────────────────────────────────────────┐
│                  ENHANCED PARALLEL PIPELINE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Video ──▶ RTMO ──▶ Gap ──▶ [Norm + Smooth] PARALLEL            │
│  (5.6s)   (5.6s) (0.8s)        (0.3s + 0.5s = 0.5s)            │
│                                   │                             │
│                                   ├─▶ [3D Lift + Phase] PARALLEL│
│                                   │  (1.5s + 0.8s = 1.5s)      │
│                                   │       │                     │
│                                   │       ├─▶ [Metrics + Ref    │
│                                   │       │     + Physics] PARA. │
│                                   │       │  (1.2s + I/O + 0.9s  │
│                                   │       │   = 1.2s)            │
│                                   │       │       │             │
│                                   │       │       └─▶ [DTW +     │
│                                   │       │          Recs] PARA. │
│                                   │       │       (0.9s + 0.4s   │
│                                   │       │        = 0.9s)       │
│                                   │       │                     │
│  TOTAL TIME: ~10.0s (17% reduction)                             │
│  PARALLEL STAGES: 8                                             │
└─────────────────────────────────────────────────────────────────┘
```

### MULTI-GPU ARCHITECTURE (2 GPUs)

```
┌─────────────────────────────────────────────────────────────────┐
│                   MULTI-GPU PIPELINE (2 GPUs)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Video ──▶ Split ──▶ [GPU 0: Frames 0-181] ──┐                │
│           Chunks    [GPU 1: Frames 182-363]  │                 │
│                     (5.6s / 2 = 2.8s)        │                 │
│                                               │                 │
│                                               ├─▶ Merge         │
│                                               │  (0.1s)         │
│                                               │                 │
│                                               └─▶ [Rest of      │
│                                                  pipeline]      │
│                                                  (4.2s)         │
│                                                                   │
│  TOTAL TIME: ~7.1s (2 GPUs)                                      │
│  SPEEDUP: 1.7x                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### BATCH PROCESSING (4 Workers)

```
┌─────────────────────────────────────────────────────────────────┐
│                  BATCH PROCESSING (4 Videos)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Video 1 ──▶ Worker 1 ──▶ Done (12s)                           │
│  Video 2 ──▶ Worker 2 ──▶ Done (12s)                           │
│  Video 3 ──▶ Worker 3 ──▶ Done (12s)                           │
│  Video 4 ──▶ Worker 4 ──▶ Done (12s)                           │
│                                                                   │
│  TOTAL TIME: 12s (4 videos processed in parallel)               │
│  SPEEDUP: 4.0x                                                  │
│  vs. Sequential: 48s → 12s                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Parallelization Strategy Comparison

### Strategy 1: AsyncIO (Current Implementation)

**What:**
- Use `asyncio.create_task` for independent stages
- Use `asyncio.to_thread` for CPU-bound operations

**Best for:**
- I/O-bound operations (API calls, file I/O)
- Low-overhead concurrency

**Limitations:**
- Single-threaded (no true parallelism)
- Not ideal for CPU-bound NumPy/ONNX

**Current usage:**
- 3D lifting + blade detection
- Phase detection
- Metrics + reference loading

---

### Strategy 2: Enhanced Async (Proposed)

**What:**
- Parallelize all independent CPU-bound stages
- Use `asyncio.to_thread` for NumPy operations
- Parallelize DTW + recommendations

**Best for:**
- Mixed I/O + CPU-bound workloads
- Single video processing

**Speedup:**
- 12-17% for single video
- Low risk

**Implementation:**
```python
# Parallel normalization + smoothing
norm_future = asyncio.to_thread(normalizer.normalize, poses)
smooth_future = asyncio.to_thread(smoother.smooth, poses)
normalized, smoothed = await asyncio.gather(norm_future, smooth_future)

# Parallel metrics + reference + physics
metrics_future = asyncio.create_task(compute_metrics_async(...))
ref_future = asyncio.create_task(load_reference_async(...))
physics_future = asyncio.create_task(compute_physics_async(...))
metrics, ref, physics = await asyncio.gather(metrics_future, ref_future, physics_future)

# Parallel DTW + recommendations
dtw_future = asyncio.create_task(compute_dtw_async(...))
rec_future = asyncio.to_thread(recommender.recommend, metrics)
dtw, recs = await asyncio.gather(dtw_future, rec_future)
```

---

### Strategy 3: Multiprocessing (Batch Processing)

**What:**
- Use `ProcessPoolExecutor` for true parallelism
- Each video processed in separate process
- Bypasses GIL

**Best for:**
- CPU-bound workloads
- Multiple videos
- Training data preprocessing

**Speedup:**
- Linear with worker count (4x for 4 workers)

**Implementation:**
```python
from concurrent.futures import ProcessPoolExecutor, as_completed

with ProcessPoolExecutor(max_workers=4) as executor:
    futures = {
        executor.submit(analyze_video, video): video
        for video in video_paths
    }
    results = {
        futures[f]: f.result()
        for f in as_completed(futures)
    }
```

---

### Strategy 4: Multi-GPU Extraction

**What:**
- Split video into chunks
- Process each chunk on separate GPU
- Merge results

**Best for:**
- Servers with multiple GPUs
- Long videos (>1000 frames)

**Speedup:**
- Near-linear with GPU count (2x for 2 GPUs)

**Implementation:**
```python
if device_config.num_gpus > 1:
    extractor = MultiGPUPoseExtractor(config=device_config.multi_gpu_config)
else:
    extractor = PoseExtractor(device=device_config.device)
```

---

## Performance Comparison Matrix

| Scenario | Current | Priority 1 | Priority 1+3 | Priority 2 (4 workers) |
|----------|---------|------------|--------------|------------------------|
| **Single video** | 12.0s | 10.0s (1.2x) | 5.0s (2.4x) | N/A |
| **4 videos** | 48.0s | 40.0s (1.2x) | 20.0s (2.4x) | 12.0s (4.0x) |
| **10 videos** | 120.0s | 100.0s (1.2x) | 50.0s (2.4x) | 30.0s (4.0x) |
| **Speedup** | 1.0x | 1.2x | 2.4x | 4.0x (batch) |

**Notes:**
- Priority 1 = Enhanced async pipeline
- Priority 3 = Multi-GPU integration (requires 2 GPUs)
- Priority 2 = Batch processing (requires 4 CPU cores)

---

## Dependency Graph (Critical Path)

```
               ┌─▶ 3D Lift ─┐
               │             ├─▶ Physics ─┐
Video ──▶ RTMO │             │            │
     └─▶ Gap ──┴─▶ Phase ───┴─▶ Metrics ─┴─▶ DTW ──▶ Recs
              │               │
              └─▶ Norm ──────┴─▶ Ref Load
                   │
                   └─▶ Smooth
```

**Critical Path (Longest):**
```
Video → RTMO (5.6s) → Gap (0.8s) → Norm (0.3s) → Smooth (0.5s)
    → Phase (0.8s) → Metrics (1.2s) → DTW (0.9s) → Recs (0.4s)
    = 10.5s (optimized)
```

**Parallelizable Branches:**
1. **3D Lift** (1.5s) - Can run with Phase
2. **Physics** (0.9s) - Can run with Metrics
3. **Ref Load** (I/O) - Can run with Metrics
4. **Norm + Smooth** (0.8s) - Can run together

**Total Parallel Savings:**
- Norm + Smooth: 0.8s → 0.5s (0.3s saved)
- 3D Lift + Phase: 2.3s → 1.5s (0.8s saved)
- Metrics + Ref + Physics: 2.1s → 1.2s (0.9s saved)
- DTW + Recs: 1.3s → 0.9s (0.4s saved)

**Total: 2.4s saved (20% reduction)**

---

## Implementation Complexity

| Priority | Time | Complexity | Risk | Files |
|----------|------|------------|------|-------|
| **Priority 1** | 1-2 days | Low | Low | 1 file (~200 lines) |
| **Priority 2** | 3-4 days | Medium | Medium | 2 files (~300 lines) |
| **Priority 3** | 2-3 days | Medium | Medium | 2 files (~150 lines) |
| **Priority 4** | 2-3 days | Low | Low | 2 files (~100 lines) |
| **Priority 5** | 1-2 days | Low | Low | 2 files (~80 lines) |

---

**End of Comparison**
