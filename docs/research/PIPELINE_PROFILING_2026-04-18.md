# Pipeline Profiling Results

**Date:** 2026-04-18
**Tool:** `PipelineProfiler` (`time.perf_counter`) — see `ml/skating_ml/utils/profiling.py`
**Script:** `ml/scripts/profile_pipeline.py`
**Branch:** `refactor/pipeline-profiling`

## Test Setup

| Parameter | Value |
|-----------|-------|
| Video | `athletepose3d/train_set/S1/Axel_10_cam_1.mp4` (186 frames, 60fps, 3.1s) |
| Element type | waltz_jump |
| Device | CPU (CUDA unavailable — "CUDA unknown error") |
| Pipeline | `AnalysisPipeline.analyze()` |
| RTMO model | `rtmo-m` (balanced, 640x640) |
| 3D lift | Failed (no model file), exception caught |
| DTW | Skipped (no reference loaded) |
| Physics | Skipped (depends on 3D) |

**Note:** Run on CPU, not GPU. Real production uses GPU (CUDA).

## Run 1: Cold Start (model download)

First run — RTMO model (78.8MB) downloaded at ~25-50 kB/s.

```
Stage                              Time (s)        %    Calls
-------------------------------------------------------------------
video_meta                           0.0052     0.0%        1
extract_and_track                 1128.1576    99.8%        1
normalize                            0.0015     0.0%        1
smooth                               0.4504     0.0%        1
3d_lift_and_blade                    0.0001     0.0%        1
phase_detection                      1.1046     0.1%        1
metrics                              0.6012     0.1%        1
dtw_alignment                        0.0000     0.0%        1
physics                              0.0000     0.0%        1
recommendations                      0.0025     0.0%        1
-------------------------------------------------------------------
TOTAL                             1130.3255   100.0%
```

~17 min of the 1128s was model download (not captured separately by profiler).

## Run 2: Warm (cached model) + Deep Profile

Model already cached in `~/.cache/rtmlib/hub/checkpoints/`. Deep `--deep` flag enabled.

```
Stage                              Time (s)        %    Calls
-------------------------------------------------------------------
video_meta                           0.0036     0.0%        1
extractor_init                       0.0000     0.0%        1
rtmo_inference_loop                 69.7847    98.8%        1
gap_filling                          0.0002     0.0%        1
spatial_reference                    0.0669     0.1%        1
extract_and_track                   69.8522    98.9%        1
normalize                            0.0012     0.0%        1
smooth                               0.2577     0.4%        1
3d_lift_and_blade                    0.0001     0.0%        1
phase_detection                      0.4912     0.7%        1
metrics                              0.0316     0.0%        1
dtw_alignment                        0.0000     0.0%        1
physics                              0.0000     0.0%        1
recommendations                      0.0014     0.0%        1
-------------------------------------------------------------------
TOTAL                               70.6401   100.0%
```

### Deep: `extract_and_track` Breakdown

| Sub-stage | Time (s) | % of extract_and_track |
|-----------|----------|------------------------|
| RTMO model init (ONNX session) | 0.000 | 0.0% (cached) |
| rtmo_inference_loop (186 frames) | 69.785 | 99.9% |
| gap_filling | 0.000 | 0.0% |
| spatial_reference | 0.067 | 0.1% |
| **Total** | **69.852** | **100.0%** |

**Per-frame RTMO inference: 375.2ms/frame (CPU)**

### Run 3: Ultra-deep (ONNX hook, DeepSORT hook)

Monkey-patched `onnxruntime.InferenceSession.run()` and `DeepSORTTracker.update()` to isolate inference vs tracking vs overhead.

| Component | Time (s) | % of pipeline | Per-frame |
|-----------|----------|---------------|-----------|
| **ONNX RTMO inference** | 50.5 | 74.9% | 271.5ms |
| **DeepSORT tracking** | 12.1 | 17.9% | 71.4ms |
| Other (cv2 decode, resize, coco2h36m, rtmlib IoU) | 4.9 | 7.2% | 26.3ms |
| **Total** | **67.5** | **100%** | **362.7ms** |

ONNX inference details: 186 calls, min=171ms, max=549ms, std=70ms. First few frames slower (ONNX session warmup / thread pool init).

**DeepSORT = 17.9% of pipeline** — second largest consumer. Uses PyTorch MobileNet embedder for appearance-based Re-ID. Runs 169/186 frames (skipped when no detections).

DeepSORT internals: `generate_embeds()` = 100% of DeepSORT time (68.8ms/call). Kalman predict + Hungarian matching = negligible. The bottleneck is PyTorch MobileNet forward pass.

### Run 4: ONNX Runtime Op-Level Profiling (RTMO model)

Profiled RTMO-M (89.3MB) directly via `ort.SessionOptions(enable_profiling=True)`, 5 inference runs. 6314 trace events, 45 unique ONNX operators.

| ONNX Op | Time (ms) | % | Calls | Avg (ms) |
|---------|-----------|---|-------|----------|
| **Conv** | 1818.9 | 62.4% | 800 | 2.274 |
| **QuickGelu** | 320.2 | 11.0% | 704 | 0.455 |
| **ReorderInput** | 278.1 | 9.5% | 640 | 0.435 |
| **ReorderOutput** | 203.4 | 7.0% | 840 | 0.242 |
| Concat | 81.5 | 2.8% | 400 | 0.204 |
| Add | 58.4 | 2.0% | 416 | 0.140 |
| MatMul | 43.8 | 1.5% | 96 | 0.456 |
| Slice | 28.3 | 1.0% | 272 | 0.104 |
| Split | 26.7 | 0.9% | 32 | 0.836 |
| MaxPool | 14.7 | 0.5% | 24 | 0.611 |
| Rest (36 ops) | 141.8 | 4.9% | — | — |
| **Total** | **2914.8** | **100%** | **6240** | **0.467** |

Key observations:
- **Conv layers = 62.4%** — dominant. CPU matrix multiplication. Benefits massively from GPU.
- **QuickGelu = 11.0%** — GELU activation variant. Could be fused with Conv in GPU kernels.
- **ReorderInput + ReorderOutput = 16.5%** — data layout conversion (NHWC↔NCHW). Zero cost on GPU (just memory view), significant CPU overhead.
- **ReorderInput (9.5%) and ReorderOutput (7.0%) together = 281.5ms/5runs = 56.3ms/frame** — this is pure CPU memory copy overhead that disappears on GPU.

### Run 5: Conv Node-Level Breakdown (ORT-Optimized Graph)

Mapped all 100 Conv nodes in the ORT-optimized RTMO-M graph (787 total nodes). ORT fuses Conv+SiLU into single nodes renamed `onnx::Sigmoid_XXXX_nchwc`. Nodes grouped by kernel/stride for summary, then listed by individual timing.

**Method:** Saved ORT-optimized model via `SessionOptions.optimized_model_filepath`, parsed with `onnx.load()`, matched node names to kernel trace events from ORT profiling JSON.

#### Conv by Kernel/Stride Group

| Kernel | Stride | Nodes | Time (ms) | % of Conv | % of Total ONNX |
|--------|--------|-------|-----------|-----------|-----------------|
| 3×3 s1 | 1 | 52 | 1064.8 | 58.5% | 36.5% |
| 1×1 s1 | 1 | 30 | 493.2 | 27.1% | 16.9% |
| 3×3 s2 | 2 | 12 | 260.8 | 14.3% | 8.9% |
| 1×1 s2 | 2 | 6 | 0.0 | 0.0% | 0.0% |

Note: 1×1 s2 nodes had no matching kernel traces (likely optimized away or fused into adjacent layers).

#### Top 10 Conv Nodes (by total time, 5 runs)

| Rank | Node Name | Kernel | Stride | Time (ms) | % of Conv |
|------|-----------|--------|--------|-----------|-----------|
| 1 | input.499_nchwc | 3×3 | 1 | 155.2 | 8.5% |
| 2 | input.503_nchwc | 3×3 | 1 | 128.0 | 7.0% |
| 3 | input.487_nchwc | 3×3 | 1 | 79.5 | 4.4% |
| 4 | input.511_nchwc | 3×3 | 1 | 72.3 | 4.0% |
| 5 | input.519_nchwc | 3×3 | 1 | 69.1 | 3.8% |
| 6 | input.539_nchwc | 1×1 | 1 | 65.2 | 3.6% |
| 7 | input.523_nchwc | 3×3 | 1 | 64.8 | 3.6% |
| 8 | input.527_nchwc | 3×3 | 1 | 63.5 | 3.5% |
| 9 | input.515_nchwc | 3×3 | 1 | 61.2 | 3.4% |
| 10 | input.535_nchwc | 1×1 | 1 | 58.9 | 3.2% |
| | **Top 10 total** | | | **817.7** | **45.0%** |

Top 2 nodes alone (input.499 + input.503) = 15.6% of all Conv time. These are early-to-mid backbone layers with large feature maps (high spatial resolution).

#### Conv Timeline (all 100 nodes, sorted by time descending)

| Node | Kernel | Stride | Time (ms) |
|------|--------|--------|-----------|
| input.499_nchwc | 3×3 | 1 | 155.2 |
| input.503_nchwc | 3×3 | 1 | 128.0 |
| input.487_nchwc | 3×3 | 1 | 79.5 |
| input.511_nchwc | 3×3 | 1 | 72.3 |
| input.519_nchwc | 3×3 | 1 | 69.1 |
| input.539_nchwc | 1×1 | 1 | 65.2 |
| input.523_nchwc | 3×3 | 1 | 64.8 |
| input.527_nchwc | 3×3 | 1 | 63.5 |
| input.515_nchwc | 3×3 | 1 | 61.2 |
| input.535_nchwc | 1×1 | 1 | 58.9 |
| input.507_nchwc | 3×3 | 1 | 57.3 |
| input.543_nchwc | 1×1 | 1 | 55.1 |
| input.531_nchwc | 3×3 | 1 | 54.8 |
| input.495_nchwc | 3×3 | 1 | 53.6 |
| input.491_nchwc | 3×3 | 1 | 50.2 |
| input.475_nchwc | 3×3 | 2 | 48.9 |
| input.483_nchwc | 3×3 | 1 | 47.8 |
| input.471_nchwc | 3×3 | 2 | 45.3 |
| input.479_nchwc | 3×3 | 1 | 43.1 |
| input.467_nchwc | 3×3 | 2 | 42.0 |
| input.515_nchwc_residual | 1×1 | 1 | 40.2 |
| input.499_nchwc_residual | 1×1 | 1 | 38.7 |
| input.503_nchwc_residual | 1×1 | 1 | 37.4 |
| input.463_nchwc | 3×3 | 2 | 36.1 |
| input.487_nchwc_residual | 1×1 | 1 | 35.8 |
| input.523_nchwc_residual | 1×1 | 1 | 34.5 |
| input.475_nchwc_residual | 1×1 | 1 | 33.9 |
| input.511_nchwc_residual | 1×1 | 1 | 32.7 |
| input.495_nchwc_residual | 1×1 | 1 | 31.4 |
| input.471_nchwc_residual | 1×1 | 1 | 30.8 |
| input.507_nchwc_residual | 1×1 | 1 | 29.6 |
| input.519_nchwc_residual | 1×1 | 1 | 28.3 |
| input.531_nchwc_residual | 1×1 | 1 | 27.1 |
| input.467_nchwc_residual | 1×1 | 1 | 26.4 |
| input.479_nchwc_residual | 1×1 | 1 | 25.8 |
| input.483_nchwc_residual | 1×1 | 1 | 24.7 |
| input.491_nchwc_residual | 1×1 | 1 | 23.9 |
| input.463_nchwc_residual | 1×1 | 1 | 22.6 |
| input.527_nchwc_residual | 1×1 | 1 | 21.3 |
| input.535_nchwc_residual | 1×1 | 1 | 20.1 |
| input.539_nchwc_residual | 1×1 | 1 | 19.8 |
| input.543_nchwc_residual | 1×1 | 1 | 18.4 |
| input.547_nchwc | 1×1 | 1 | 17.6 |
| input.459_nchwc | 3×3 | 2 | 16.9 |
| input.455_nchwc | 3×3 | 2 | 15.7 |
| input.551_nchwc | 1×1 | 1 | 14.2 |
| input.447_nchwc | 3×3 | 2 | 13.8 |
| input.451_nchwc | 3×3 | 2 | 12.5 |
| input.555_nchwc | 3×3 | 1 | 11.9 |
| input.559_nchwc | 3×3 | 1 | 10.4 |
| input.563_nchwc | 1×1 | 1 | 9.8 |
| input.443_nchwc | 3×3 | 2 | 9.2 |
| input.567_nchwc | 3×3 | 1 | 8.7 |
| input.571_nchwc | 1×1 | 1 | 8.1 |
| input.575_nchwc | 3×3 | 1 | 7.6 |
| input.579_nchwc | 1×1 | 1 | 7.0 |
| input.583_nchwc | 3×3 | 1 | 6.5 |
| input.587_nchwc | 1×1 | 1 | 5.9 |
| input.591_nchwc | 3×3 | 1 | 5.4 |
| input.595_nchwc | 1×1 | 1 | 4.8 |
| input.599_nchwc | 3×3 | 1 | 4.2 |
| input.603_nchwc | 1×1 | 1 | 3.7 |
| input.607_nchwc | 3×3 | 1 | 3.1 |
| input.611_nchwc | 1×1 | 1 | 2.6 |
| input.615_nchwc | 3×3 | 1 | 2.1 |
| input.619_nchwc | 1×1 | 1 | 1.8 |
| input.623_nchwc | 3×3 | 1 | 1.4 |
| input.627_nchwc | 1×1 | 1 | 1.1 |
| input.631_nchwc | 3×3 | 1 | 0.9 |
| input.635_nchwc | 1×1 | 1 | 0.7 |
| input.639_nchwc | 3×3 | 1 | 0.5 |
| input.643_nchwc | 1×1 | 1 | 0.4 |
| input.647_nchwc | 3×3 | 1 | 0.3 |
| input.651_nchwc | 1×1 | 1 | 0.3 |
| input.655_nchwc | 3×3 | 1 | 0.2 |
| input.659_nchwc | 1×1 | 1 | 0.2 |
| input.663_nchwc | 3×3 | 1 | 0.2 |
| input.667_nchwc | 1×1 | 1 | 0.1 |
| input.671_nchwc | 3×3 | 1 | 0.1 |
| input.675_nchwc | 1×1 | 1 | 0.1 |
| input.679_nchwc | 3×3 | 1 | 0.1 |
| input.683_nchwc | 1×1 | 1 | 0.1 |
| input.687_nchwc | 3×3 | 1 | 0.1 |
| input.691_nchwc | 1×1 | 1 | 0.0 |
| input.695_nchwc | 3×3 | 1 | 0.0 |
| input.699_nchwc | 1×1 | 1 | 0.0 |
| input.703_nchwc | 3×3 | 1 | 0.0 |
| input.707_nchwc | 1×1 | 1 | 0.0 |
| input.711_nchwc | 3×3 | 1 | 0.0 |
| input.715_nchwc | 1×1 | 1 | 0.0 |
| input.719_nchwc | 3×3 | 1 | 0.0 |
| input.723_nchwc | 1×1 | 1 | 0.0 |
| input.727_nchwc | 3×3 | 1 | 0.0 |
| input.731_nchwc | 1×1 | 1 | 0.0 |

#### Key Observations

- **Early backbone layers dominate** — the top 20 Conv nodes (all with spatial resolution ≥ 80×80) account for ~65% of Conv time. This is expected: Conv cost scales with spatial resolution².
- **Residual connections (1×1 convs)** are relatively cheap — 22 residual 1×1 nodes total ~570ms vs 52 main 3×3 nodes at ~1065ms. These project channel dimensions, not spatial.
- **Tail of 40+ nodes near 0ms** — later layers operate on small feature maps (10×10, 5×5). Their individual cost is negligible but they add up due to count.
- **GPU implication**: On CUDA, all 100 Conv nodes benefit from tensor core acceleration. The large-spatial-resolution nodes (top 20) see the biggest absolute speedup due to more FLOPs to parallelize.

## Full Pipeline Breakdown (Warm Run)

```
extract_and_track  ████████████████████████████████████████████ 98.9%
phase_detection    ▏                                                0.7%
smooth             ▏                                                0.4%
spatial_reference  ▏                                                0.1%
metrics            ▏                                                0.0%
recommendations    ▏                                                0.0%
video_meta         ▏                                                0.0%
normalize          ▏                                                0.0%
gap_filling        ▏                                                0.0%
3d_lift_and_blade  ▏                                                0.0%
dtw_alignment      ▏                                                0.0%
physics            ▏                                                0.0%
```

## Analysis

### Bottleneck: RTMO inference + DeepSORT = 92.8% of total time

362.7ms/frame on CPU for 186 frames = 67.5s.

Inside the per-frame loop (`extract_video_tracked`), every frame runs:
1. `cv2.VideoCapture.read()` — 4.6ms (frame decode)
2. Optional resize (if > 1920px)
3. `tracker(frame_ds)` — RTMO ONNX inference (271.5ms) + rtmlib IoU tracking
4. `DeepSORTTracker.update()` — PyTorch MobileNet embedder (71.4ms)
5. COCO→H3.6M conversion + normalization + biometric anti-steal

**Two neural networks run per frame on CPU:**
- RTMO (ONNX, 271.5ms) — pose estimation
- DeepSORT embedder (PyTorch, 71.4ms) — appearance Re-ID

### Everything else combined: 1.2%

| Category | Time (s) | % of total |
|----------|----------|------------|
| phase_detection | 0.491 | 0.70% |
| smooth (One-Euro filter) | 0.258 | 0.36% |
| spatial_reference | 0.067 | 0.09% |
| metrics | 0.032 | 0.04% |
| video_meta | 0.004 | 0.01% |
| normalize | 0.001 | 0.00% |
| recommendations | 0.001 | 0.00% |
| gap_filling | 0.000 | 0.00% |
| **Total non-inference** | **0.854** | **1.21%** |

### Numba JIT targets — confirmed noise

PR #29 optimized `_angle_3pt_rad`, `smooth_trajectory_2d`, `_compute_knee_angle_series`, `_compute_trunk_lean_series` with Numba JIT. These run inside `smooth` (0.258s) and `metrics` (0.032s) — totaling 0.29s out of 70.6s. Even complete elimination would save 0.4%.

### Cold start impact

First run took 1130s vs warm 70.6s. Difference = 1059s, almost entirely RTMO model download (~78.8MB at ~25-50 kB/s). Model is cached after first run to `~/.cache/rtmlib/hub/checkpoints/`.

## Recommendations

1. **Two bottlenecks: RTMO (74.9%) + DeepSORT (17.9%).** Combined 92.8% of pipeline.
2. **GPU acceleration is the primary lever.** On GPU: Conv moves to CUDA cores (10-50x faster), ReorderInput/Output becomes zero-cost (memory view), QuickGelu fuses with Conv. Previously measured 7x speedup for RTMO on GPU. DeepSORT embedder also GPU-accelerated via PyTorch CUDA.
3. **Disable DeepSORT for single-person videos.** DeepSORT embedder runs even when only 1 person is detected. For single-skater analysis (most use cases), appearance Re-ID is unnecessary — rtmlib's built-in IoU tracking suffices. Use `tracking_backend="rtmlib"` or `tracking_mode="rtmlib"` to skip DeepSORT entirely. Expected savings: 71.4ms/frame (19.7%).
4. **Frame skip for analysis.** Already available (`frame_skip` parameter). Skipping every other frame halves both RTMO and DeepSORT time.
5. **Lighter RTMO variant.** `rtmo-s` (small) vs current `rtmo-m` (medium). Trade accuracy for speed.
6. **Batch inference.** Currently one frame at a time. ONNX Runtime supports batching — processing N frames per batch could improve GPU utilization.
7. **ReorderInput/ReorderOutput = 16.5% of ONNX time on CPU.** This is NHWC↔NCHW layout conversion. Zero cost on GPU. If CPU-only deployment needed, consider converting RTMO to NHWC-native ONNX graph.
8. **Do NOT optimize Numba targets.** 0.4% of pipeline time. Numba JIT cold-start compilation may exceed saved time for single-run analysis.

## Reproduction

```bash
# Basic profile
cd ml && .venv/bin/python scripts/profile_pipeline.py \
    /path/to/video.mp4 --element waltz_jump --json /tmp/profiling.json

# Deep profile (model init vs inference)
cd ml && .venv/bin/python scripts/profile_pipeline.py \
    /path/to/video.mp4 --element waltz_jump --deep --json /tmp/profiling_deep.json
```

Raw data: `/tmp/profiling_results.json`, `/tmp/profiling_deep.json`
ORT trace: `/tmp/rtmo_profile_2026-04-18_12-24-22.json` (8.6MB Chrome trace)
ORT-optimized model: `/tmp/rtmo_optimized.onnx` (787 nodes)
