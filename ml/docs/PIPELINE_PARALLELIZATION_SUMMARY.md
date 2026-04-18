# Pipeline Parallelization - Executive Summary

## TL;DR

**Current:** 12s for 14.5s video (364 frames)
**Potential:** 6-8s (30-50% reduction) with proposed changes
**Highest Impact:** Enhanced async pipeline (12-17% speedup, 1-2 days work)

---

## Key Findings

### 1. Existing Async is Incomplete

**What's parallelized:**
- ✅ 3D lifting + blade detection
- ✅ Phase detection
- ✅ Metrics + reference loading

**What's NOT parallelized:**
- ❌ Normalization + smoothing (sequential)
- ❌ DTW + recommendations (sequential)
- ❌ Physics calculations (sequential)

**Impact:** Missing 12-17% potential speedup

---

### 2. Bottleneck Analysis

| Stage | Time | % | Parallelizable? | Action |
|-------|------|---|-----------------|--------|
| RTMO Inference | 5.6s | 47% | ⚠️ GPU-bound | Already fast |
| Gap Filling | 0.8s | 7% | ✅ Yes | Parallelize |
| Normalization | 0.3s | 3% | ✅ Yes | Parallelize |
| Smoothing | 0.5s | 4% | ✅ Yes | Already Numba-opt |
| 3D Lifting | 1.5s | 13% | ✅ Yes | Already parallel |
| Phase Detection | 0.8s | 7% | ✅ Yes | Already parallel |
| Metrics | 1.2s | 10% | ✅ Yes | Parallelize |
| DTW | 0.9s | 8% | ✅ Yes | Parallelize |
| Recommendations | 0.4s | 3% | ✅ Yes | Parallelize |

**Total parallelizable potential:** 6.4s (53% of runtime)

---

### 3. Multi-GPU Extraction Exists But Not Integrated

**File:** `ml/skating_ml/pose_estimation/multi_gpu_extractor.py`

**Status:** Fully implemented, not wired into main pipeline

**Impact:** 2-4x speedup with 2-4 GPUs

**Action:** 2-3 days to integrate

---

## Proposed Solutions (Ranked by Impact)

### 🥇 Priority 1: Enhanced Async Pipeline

**Time:** 1-2 days
**Impact:** 12-17% speedup
**Risk:** LOW

**Changes:**
```python
# Parallelize normalization + smoothing
normalized_future = asyncio.to_thread(normalizer.normalize, poses)
smoothed_future = asyncio.to_thread(smoother.smooth, poses)
normalized = await normalized_future
smoothed = await smoothed_future

# Parallelize metrics + reference + physics
metrics_future = asyncio.create_task(compute_metrics_async(...))
ref_future = asyncio.create_task(load_reference_async(...))
physics_future = asyncio.create_task(compute_physics_async(...))

# Parallelize DTW + recommendations
dtw_future = asyncio.create_task(compute_dtw_async(...))
rec_future = asyncio.to_thread(recommender.recommend, metrics)
```

**Files:** `ml/skating_ml/pipeline.py` (lines 605-870)

---

### 🥈 Priority 2: Batch Processing

**Time:** 3-4 days
**Impact:** 4-16x speedup (for multiple videos)
**Risk:** MEDIUM

**Use case:** Processing 10+ videos (reference database, training data)

**Changes:**
```python
with ProcessPoolExecutor(max_workers=4) as executor:
    futures = {
        executor.submit(analyze_video, video): video
        for video in video_paths
    }
    results = [f.result() for f in as_completed(futures)]
```

**Files:** `ml/skating_ml/batch.py` (new)

---

### 🥉 Priority 3: Multi-GPU Integration

**Time:** 2-3 days
**Impact:** 2-4x speedup (with 2-4 GPUs)
**Risk:** MEDIUM

**Use case:** Servers with multiple GPUs (Vast.ai)

**Changes:**
```python
if device_config.num_gpus > 1:
    extractor = MultiGPUPoseExtractor(config=device_config.multi_gpu_config)
else:
    extractor = PoseExtractor(device=device_config.device)
```

**Files:** `ml/skating_ml/pipeline.py` (lines 94-174)

---

## Performance Estimates

### Single Video (364 frames, 14.5s)

| Configuration | Time | Speedup |
|---------------|------|---------|
| Current (1 GPU) | 12.0s | 1.0x |
| + Priority 1 (Enhanced async) | 10.0s | 1.2x |
| + Priority 3 (2 GPUs) | 5.0s | 2.4x |
| + All optimizations | 4.2s | 2.9x |

### Batch Processing (10 videos)

| Configuration | Time | Speedup |
|---------------|------|---------|
| Sequential | 120s | 1.0x |
| Priority 2 (4 workers) | 30s | 4.0x |
| Priority 2 (8 workers) | 15s | 8.0x |
| Priority 1 + 2 (4 workers) | 25s | 4.8x |

---

## Implementation Plan

### Week 1

- [ ] Implement Priority 1 (Enhanced Async)
- [ ] Profile and validate speedup
- [ ] Update tests

### Week 2-3

- [ ] Implement Priority 2 (Batch Processing)
- [ ] Add CLI command for batch processing
- [ ] Document usage

### Week 4

- [ ] Integrate Priority 3 (Multi-GPU)
- [ ] Test on Vast.ai (2-4 GPUs)
- [ ] Document setup

---

## Research Findings

### Best Practices

1. **AsyncIO** for I/O-bound (API calls, DB queries)
2. **Threading** for I/O-bound + some NumPy (GIL released)
3. **Multiprocessing** for CPU-bound (ONNX, heavy NumPy)

### Sources

- learnomate.org: "Parallel Processing for Data Analysis"
- testdriven.io: "Parallelism, Concurrency, and AsyncIO in Python"
- xailient.com: "Parallel Processing for Faster Video Processing"
- GitHub issue #7846: "ONNX Runtime + Multiprocessing"

### Key Takeaways

- ONNX Runtime sessions work with multiprocessing (separate sessions per process)
- NumPy releases GIL for most operations (threading can help)
- Profile before optimizing (overhead can exceed benefits)

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Async overhead > benefits | Medium | Profile before/after, only parallelize >200ms ops |
| Data races | High | Use immutable NumPy arrays, no shared state |
| Memory usage | Medium | Limit batch sizes, use generators |
| GPU memory exhaustion | High | Monitor VRAM, fall back to single GPU |
| ONNX + multiprocessing | Medium | Use ProcessPoolExecutor, test thoroughly |

---

## Next Steps

1. **Immediate:** Implement Priority 1 (1-2 days)
2. **Short-term:** Implement Priority 2 (3-4 days)
3. **Medium-term:** Integrate Priority 3 (2-3 days)
4. **Long-term:** Evaluate CPU-bound optimizations (2-3 days)

---

## Files to Modify

### Priority 1

- `ml/skating_ml/pipeline.py` (lines 605-870)

### Priority 2

- `ml/skating_ml/batch.py` (new)
- `ml/skating_ml/cli.py` (add batch command)

### Priority 3

- `ml/skating_ml/pipeline.py` (lines 94-174)
- `ml/skating_ml/device.py` (add multi-GPU config)

---

## Testing Strategy

### Unit Tests

- Test parallel stages produce same results as sequential
- Test error handling in parallel context
- Test memory usage doesn't leak

### Integration Tests

- Test full pipeline with async enabled
- Test batch processing with multiple videos
- Test multi-GPU extraction (if hardware available)

### Performance Tests

- Benchmark before/after for single video
- Benchmark batch processing scaling
- Profile to identify new bottlenecks

---

## Success Criteria

- [ ] Single video processing reduced by 12-17%
- [ ] Batch processing scales linearly with workers
- [ ] Multi-GPU extraction works (2-4x speedup)
- [ ] All tests pass
- [ ] No memory leaks
- [ ] Documentation updated

---

**End of Summary**
