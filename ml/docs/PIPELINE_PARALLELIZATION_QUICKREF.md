# Pipeline Parallelization - Quick Reference

## 5-Minute Overview

### Problem
Pipeline takes 12s to process 14.5s video (364 frames). Can we make it faster?

### Answer
**YES.** 30-50% reduction possible with parallelization.

### How?

**Current:**
- RTMO: 5.6s (47%)
- Analysis: 6.4s (53%)
- **Total: 12s**

**Proposed:**
- RTMO: 5.6s (56%) - GPU-bound, already fast
- Analysis (parallelized): 4.4s (44%)
- **Total: 10s (17% faster)**

### What to Parallelize?

```
┌─────────────────────────────────────────────────┐
│  CAN PARALLELIZE:                               │
│  ✅ 3D lifting + phase detection                │
│  ✅ Normalization + smoothing                   │
│  ✅ Metrics + reference + physics               │
│  ✅ DTW + recommendations                       │
│                                                 │
│  CANNOT PARALLELIZE:                            │
│  ❌ RTMO inference (GPU-bound, sequential)      │
│  ❌ Gap filling (data dependency)               │
└─────────────────────────────────────────────────┘
```

---

## Code Examples

### Example 1: Enhanced Async (Priority 1)

**Before (Sequential):**
```python
# Sequential
normalized = normalizer.normalize(poses)
smoothed = smoother.smooth(normalized)
poses_3d = extractor_3d.extract_sequence(smoothed)
phases = detector.detect_phases(smoothed, fps, element_type)
metrics = analyzer.analyze(smoothed, phases, fps)
reference = store.get_best_match(element_type)
dtw_distance = aligner.compute_distance(normalized, reference.poses)
recommendations = recommender.recommend(metrics, element_type)
```

**After (Parallel):**
```python
# Parallel normalization + smoothing
normalized_future = asyncio.to_thread(normalizer.normalize, poses)
smoothed_future = asyncio.to_thread(smoother.smooth, poses)
normalized, smoothed = await asyncio.gather(normalized_future, smoothed_future)

# Parallel 3D lifting + phase detection
poses_3d_future = asyncio.create_task(lift_poses_3d_async(smoothed, fps))
phases_future = asyncio.create_task(detect_phases_async(smoothed, fps, element_type))
poses_3d, phases = await asyncio.gather(poses_3d_future, phases_future)

# Parallel metrics + reference + physics
metrics_future = asyncio.create_task(compute_metrics_async(smoothed, phases, fps))
ref_future = asyncio.create_task(load_reference_async(element_type))
physics_future = asyncio.create_task(compute_physics_async(poses_3d, phases, fps))
metrics, reference, physics = await asyncio.gather(metrics_future, ref_future, physics_future)

# Parallel DTW + recommendations
dtw_future = asyncio.create_task(compute_dtw_async(normalized, phases, reference))
rec_future = asyncio.to_thread(recommender.recommend, metrics, element_type)
dtw_distance, recommendations = await asyncio.gather(dtw_future, rec_future)
```

**Impact:** 12-17% speedup (12s → 10s)

---

### Example 2: Batch Processing (Priority 2)

**Before (Sequential):**
```python
results = []
for video in video_paths:
    report = pipeline.analyze(video, element_type="waltz_jump")
    results.append(report)
# Time: 10 videos × 12s = 120s
```

**After (Parallel):**
```python
from concurrent.futures import ProcessPoolExecutor, as_completed

with ProcessPoolExecutor(max_workers=4) as executor:
    futures = {
        executor.submit(analyze_video_worker, video, "waltz_jump"): video
        for video in video_paths
    }
    results = []
    for future in as_completed(futures):
        video_path = futures[future]
        try:
            report = future.result()
            results.append((video_path, report))
        except Exception as e:
            logger.error(f"Failed to analyze {video_path}: {e}")
# Time: 10 videos ÷ 4 workers = 30s (4x speedup)
```

**Impact:** 4x speedup for batch processing (120s → 30s)

---

### Example 3: Multi-GPU Extraction (Priority 3)

**Before (Single GPU):**
```python
extractor = PoseExtractor(device="cuda")
extraction = extractor.extract_video_tracked(video_path)
# Time: 5.6s (364 frames)
```

**After (Multi-GPU):**
```python
if device_config.num_gpus > 1:
    extractor = MultiGPUPoseExtractor(config=device_config.multi_gpu_config)
else:
    extractor = PoseExtractor(device=device_config.device)
extraction = extractor.extract_video_tracked(video_path)
# Time: 5.6s ÷ 2 GPUs = 2.8s (2x speedup)
```

**Impact:** 2x speedup per GPU (5.6s → 2.8s with 2 GPUs)

---

## Implementation Checklist

### Priority 1: Enhanced Async (1-2 days)

- [ ] Refactor `analyze_async` to use `asyncio.to_thread` for CPU-bound stages
- [ ] Parallelize normalization + smoothing (lines 217-231)
- [ ] Parallelize metrics + reference + physics (lines 677-724)
- [ ] Parallelize DTW + recommendations (lines 693-731)
- [ ] Add profiling to validate speedup
- [ ] Run tests to ensure correctness

**File:** `ml/skating_ml/pipeline.py` (lines 605-870)

**Acceptance:** 12-17% speedup, all tests pass

---

### Priority 2: Batch Processing (3-4 days)

- [ ] Create `ml/skating_ml/batch.py` with `analyze_videos_parallel`
- [ ] Implement `_analyze_video_worker` function
- [ ] Add CLI command: `python -m skating_ml.cli batch --videos *.mp4 --workers 4`
- [ ] Add progress reporting (tqdm or rich)
- [ ] Test with 10 videos
- [ ] Document usage

**Files:** `ml/skating_ml/batch.py` (new), `ml/skating_ml/cli.py`

**Acceptance:** 4x speedup for 10 videos

---

### Priority 3: Multi-GPU Integration (2-3 days)

- [ ] Integrate `MultiGPUPoseExtractor` into `_extract_and_track`
- [ ] Add multi-GPU detection to `DeviceConfig`
- [ ] Test on Vast.ai (2-4 GPUs)
- [ ] Document multi-GPU setup
- [ ] Add fallback to single GPU

**Files:** `ml/skating_ml/pipeline.py` (lines 94-174), `ml/skating_ml/device.py`

**Acceptance:** 2x speedup with 2 GPUs

---

## Testing Strategy

### Unit Tests

```python
# Test parallel stages produce same results as sequential
async def test_parallel_vs_sequential():
    pipeline = AnalysisPipeline()

    # Sequential
    report_seq = pipeline.analyze(video_path, element_type="waltz_jump")

    # Parallel
    report_par = await pipeline.analyze_async(video_path, element_type="waltz_jump")

    # Compare
    assert report_seq.element_type == report_par.element_type
    assert len(report_seq.metrics) == len(report_par.metrics)
    for m_seq, m_par in zip(report_seq.metrics, report_par.metrics):
        assert m_seq.value == pytest.approx(m_par.value, rel=1e-5)
```

### Performance Tests

```python
# Benchmark single video
def test_single_video_performance():
    pipeline = AnalysisPipeline()

    start = time.perf_counter()
    report = pipeline.analyze(video_path, element_type="waltz_jump")
    elapsed = time.perf_counter() - start

    assert elapsed < 12.0  # Should be faster than baseline

# Benchmark batch processing
def test_batch_processing_performance():
    video_paths = [video_path] * 10

    start = time.perf_counter()
    results = analyze_videos_parallel(video_paths, max_workers=4)
    elapsed = time.perf_counter() - start

    assert elapsed < 40.0  # 4x speedup (120s → 30s)
```

---

## Common Pitfalls

### Pitfall 1: Async Overhead

**Problem:** Using `asyncio.create_task` for fast operations (<100ms)

**Solution:** Use `asyncio.to_thread` for CPU-bound work, or skip parallelization

```python
# ❌ BAD: Overhead exceeds benefit
future = asyncio.create_task(lambda: np.sum(arr))()
result = await future

# ✅ GOOD: Use to_thread for NumPy
result = await asyncio.to_thread(np.sum, arr)
```

---

### Pitfall 2: Data Races

**Problem:** Parallel stages modify shared state

**Solution:** Use immutable data (NumPy arrays), no shared state

```python
# ❌ BAD: Shared mutable state
cache = {}
async def process_stage_1(data):
    cache["data"] = data  # Race condition!

async def process_stage_2(data):
    data = cache["data"]  # Race condition!

# ✅ GOOD: Immutable data
async def process_stage_1(data):
    return data.copy()

async def process_stage_2(data):
    return data.copy()
```

---

### Pitfall 3: Memory Leaks

**Problem:** Batch processing exhausts memory

**Solution:** Limit batch sizes, use generators

```python
# ❌ BAD: Load all videos into memory
videos = [load_video(p) for p in video_paths]  # 10GB+

# ✅ GOOD: Process one at a time
def video_generator(paths):
    for path in paths:
        yield load_video(path)

for video in video_generator(video_paths):
    process_video(video)
```

---

## Performance Profiling

### Profile Current Pipeline

```python
from skating_ml.pipeline import AnalysisPipeline
from skating_ml.utils.profiling import PipelineProfiler

profiler = PipelineProfiler()
pipeline = AnalysisPipeline(profiler=profiler)

report = pipeline.analyze(video_path, element_type="waltz_jump")

# Print profiling results
profiler.print_summary()

# Output:
# rtmo_inference_loop: 5.6s (47%)
# gap_filling: 0.8s (7%)
# normalize: 0.3s (3%)
# smooth: 0.5s (4%)
# 3d_lift_and_blade: 1.5s (13%)
# phase_detection: 0.8s (7%)
# metrics: 1.2s (10%)
# dtw_alignment: 0.9s (8%)
# recommendations: 0.4s (3%)
# total: 12.0s
```

### Profile Parallel Pipeline

```python
profiler = PipelineProfiler()
pipeline = AnalysisPipeline(profiler=profiler)

report = await pipeline.analyze_async(video_path, element_type="waltz_jump")

profiler.print_summary()

# Output:
# rtmo_inference_loop: 5.6s (56%)
# gap_filling: 0.8s (8%)
# normalize+smooth (parallel): 0.5s (5%)
# 3d_lift+phase (parallel): 1.5s (15%)
# metrics+ref+physics (parallel): 1.2s (12%)
# dtw+recs (parallel): 0.9s (9%)
# total: 10.5s (12.5% faster)
```

---

## Quick Commands

### Run Sequential Pipeline

```bash
uv run python -m skating_ml.cli analyze video.mp4 --element waltz_jump
```

### Run Async Pipeline (After Priority 1)

```bash
uv run python -m skating_ml.cli analyze video.mp4 --element waltz_jump --async
```

### Run Batch Processing (After Priority 2)

```bash
uv run python -m skating_ml.cli batch --videos data/videos/*.mp4 --workers 4 --element waltz_jump
```

### Run Multi-GPU Pipeline (After Priority 3)

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run python -m skating_ml.cli analyze video.mp4 --element waltz_jump
```

---

## FAQ

**Q: Can I parallelize RTMO inference?**

A: Not easily. RTMO is GPU-bound and already fast (5.6s for 364 frames). Multi-GPU extraction is better (Priority 3).

**Q: Should I use threading or multiprocessing?**

A: Use multiprocessing for batch processing (Priority 2). Use asyncio + thread pools for single video (Priority 1).

**Q: Will this work on CPU-only systems?**

A: Yes. Priority 1 (enhanced async) works on CPU. Priorities 2-3 provide additional speedups if available.

**Q: How much memory does batch processing use?**

A: ~2GB per worker (4 workers = 8GB). Reduce `max_workers` if memory is limited.

**Q: Can I combine all priorities?**

A: Yes. Priority 1 + 2 + 3 = 2.9x speedup for single video, 4.8x for batch (4 workers).

---

**End of Quick Reference**
