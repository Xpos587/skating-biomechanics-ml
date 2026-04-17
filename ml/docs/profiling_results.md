# Profiling Results

## Hotspots Identified

> Run `uv run python ml/scripts/profile_pipeline.py /path/to/video.mp4` to generate results.

## Expected Hotspots

Based on code analysis, likely optimization targets:

1. **Geometry functions** - `ml/skating_ml/utils/geometry.py`
   - `calculate_angle()` - called repeatedly for joint angles
   - `calculate_distance()` - used throughout metrics
   - **Action:** Apply Numba JIT

2. **Smoothing filter** - `ml/skating_ml/utils/smoothing.py`
   - `PoseSmoother.smooth_pose()` - per-frame filtering
   - OneEuro filter computation - loops over keypoints
   - **Action:** Apply Numba JIT

3. **Biomechanics metrics** - `ml/skating_ml/analysis/metrics.py`
   - Knee angle calculation - per-frame computation
   - Center of mass calculation - per-frame weighted average
   - Velocity calculation - per-frame derivatives
   - **Action:** Apply Numba JIT

## Priority Order

1. **geometry.py** - Foundation for everything else
2. **smoothing.py** - Called for every frame
3. **metrics.py** - Analysis bottlenecks

## How to Run

```bash
# Profile with CUDA (default)
uv run python ml/scripts/profile_pipeline.py /path/to/video.mp4

# Profile with CPU
uv run python ml/scripts/profile_pipeline.py /path/to/video.mp4 cpu
```
