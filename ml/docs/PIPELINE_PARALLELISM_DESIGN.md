# ML Pipeline Parallelism & Async Optimization Design

> **Branch:** `refactor/pipeline-profiling`
> **Date:** 2026-04-18
> **Scope:** Video processing pipeline performance optimization

---

## 1. Executive Summary

Five specialized research agents analyzed the ML pipeline from different angles (GPU inference, I/O, task orchestration, CPU preprocessing, DAG structure). After aggressive deduplication, **37 distinct optimization opportunities** were identified across the pipeline.

### Top-3 Highest-Impact Changes

| # | Opportunity | Why It Wins | Agents |
|---|-------------|-------------|--------|
| 1 | **Direct ONNX batch RTMO inference** | Pose extraction is 50-70% of wall time. rtmlib processes frames one-by-one internally. Calling the RTMO ONNX model directly with batched inputs eliminates per-frame kernel launch overhead. | 1, 4, 5 |
| 2 | **`detect_video_task` blocks the event loop** | Actual bug: GPU inference at `worker.py:368` runs synchronously inside an `async` function without `asyncio.to_thread()`. This freezes the arq worker for the entire detection duration (~5s). | 2, 3 |
| 3 | **Render + Analysis parallelism** | The render loop and biomechanics analysis are completely independent. Render needs only `PreparedPoses`; analysis needs `poses_norm + fps`. Running them via `ThreadPoolExecutor` saves the full analysis time (~200ms). | 4, 5 |

### Opportunity Breakdown by Category

| Category | Count | Highest-Impact Example |
|----------|-------|----------------------|
| GPU Inference | 9 | Batch RTMO, batch 3D lifter, CUDA graphs |
| I/O Pipeline | 8 | boto3->aiobotocore, producer-consumer decode, background encoding |
| Task Orchestration | 9 | Event loop blocking, async analyze, worker concurrency |
| CPU Preprocessing | 6 | Vectorize per-frame loops, parallel DTW+physics |
| Pipeline DAG | 5 | Render+analysis overlap, extras parallelism, early termination |

---

## 2. Current Pipeline DAG

### 2.1 `process_video_task` (arq worker -> Vast.ai path)

```
process_video_task (worker.py:191)
|
+-- await valkey.hset RUNNING                    [I/O: Valkey]
+-- await asyncio.to_thread(process_video_remote) [BLOCKS 60-300s]
|   +-- _get_worker_url()                        [I/O: HTTP 30s timeout]
|   +-- httpx.post /process                      [I/O: HTTP 600s timeout]
|       +-- gpu_server/server.py:
|           +-- boto3 download_file (BLOCKS!)     [I/O: R2 sync]
|           +-- process_video_pipeline()          [BLOCKS 30-120s]
|           |   (see 2.2 below)
|           +-- boto3 upload_file x3 (BLOCKS!)    [I/O: R2 sync]
|
+-- download_file() (BLOCKS!)                    [I/O: R2 sync]
+-- _sample_poses()                              [CPU: NumPy]
+-- _compute_frame_metrics()                     [CPU: Python loop, ~500 iterations]
+-- await store_result()                          [I/O: Valkey]
+-- await save_analysis_results()                [I/O: Postgres]
+-- await valkey.close()                          [I/O: Valkey]
```

### 2.2 `process_video_pipeline` (GPU server / web_helpers.py:136)

```
process_video_pipeline (web_helpers.py:136)
|
+-- prepare_poses()                               [50-70% of time]
|   +-- get_video_meta()                         [I/O: cv2.VideoCapture]
|   +-- PoseExtractor.extract_video_tracked()    [GPU: RTMO per-frame]
|   |   +-- Per-frame loop (web_helpers.py:325):
|   |       +-- cv2.VideoCapture.read()          [I/O: decode]
|   |       +-- tracker(frame)                   [GPU: RTMO ONNX, 1 call/frame]
|   |       +-- Track association (sports2d/deepsort)
|   +-- GapFiller.fill_gaps()                    [CPU: NumPy]
|   +-- PoseSmoother.smooth()                    [CPU: One-Euro filter]
|   +-- [Optional] 3D lift + CorrectiveLens      [GPU: MotionAGFormer]
|
+-- [Optional] Load ML extras (depth/flow/seg/matting/foot/inpainting)
|
+-- Render loop (web_helpers.py:325-384)         [20-40% of time]
|   +-- cv2.VideoCapture.read()                  [I/O: decode]
|   +-- pipe.render_frame()                      [CPU: OpenCV drawing]
|   +-- [Per-frame ML] depth/flow/seg/matting    [GPU: sequential per-frame]
|   +-- H264Writer.write(frame)                   [CPU: encode]
|
+-- [Sequential after render] Analysis            [~200ms, NOT on critical path]
|   +-- PhaseDetector.detect_phases()            [CPU: NumPy]
|   +-- BiomechanicsAnalyzer.analyze()           [CPU: NumPy]
|   +-- Recommender.recommend()                  [CPU: rules]
|
+-- pipe.save_exports()                           [I/O: write .npy, .csv]
```

### 2.3 `detect_video_task` (arq worker -> local GPU)

```
detect_video_task (worker.py:334)
|
+-- await valkey.hset RUNNING                     [I/O: Valkey]
+-- download_file()                               [I/O: R2 sync, BLOCKS event loop!]
+-- PoseExtractor(...)                            [GPU: loads RTMO ONNX session]
+-- extractor.preview_persons(video_path, 30)     [GPU: RTMO x30 frames, BLOCKS event loop!]
+-- render_person_preview(frame, persons)         [CPU: OpenCV]
+-- cv2.imencode(".png", annotated)              [CPU: PNG encode]
+-- await store_result()                          [I/O: Valkey]
+-- await valkey.close()                          [I/O: Valkey]
```

### 2.4 Timing Profile (from existing benchmarks)

| Stage | Wall Time (364 frames) | % of Total |
|-------|----------------------|------------|
| RTMO inference loop | 5.6s (GPU) | ~47% |
| Gap filling | <0.1s | ~1% |
| Smoothing | <0.1s | ~1% |
| 3D lifting (optional) | 1-3s | ~8-25% |
| Render loop | 3-5s | ~25-42% |
| Analysis (phases+metrics+recs) | <0.2s | ~2% |
| **Total** | **~12s** | 100% |

---

## 3. Opportunities by Category

### 3.1 GPU Inference (9 opportunities)

#### G1. Direct ONNX Batch RTMO Inference

**Description:** The `PoseExtractor` (pose_extractor.py:48) uses rtmlib's `PoseTracker`, which calls the RTMO ONNX model one frame at a time internally. The `BatchPoseExtractor` (batch_extractor.py:44) was created as a workaround but is a no-op: `_process_batch` (line 247) still loops frame-by-frame calling `tracker(frame)` per iteration (line 276). The real fix is to bypass rtmlib entirely, call the RTMO ONNX session directly with batched input tensors `(B, 3, 640, 640)`, and run a single `session.run()` call.

**Agents:** 1, 4, 5

**Impact:** 2-4x RTMO throughput. RTMO's ONNX model accepts batch input; the current per-frame approach pays kernel launch overhead ~500 times for a 15s video.

**Complexity:** Medium. Requires reverse-engineering rtmlib's preprocessing (letterbox resize, normalization) and postprocessing (NMS, keypoint decoding). Tracking logic must be decoupled from inference.

**Dependencies:** None. Can be implemented independently.

**Files:**
- `ml/skating_ml/pose_estimation/pose_extractor.py` (line 270: `tracker(frame)`)
- `ml/skating_ml/pose_estimation/batch_extractor.py` (line 276: same issue)
- `ml/skating_ml/pose_estimation/h36m.py` (COCO->H3.6M conversion)

**Validation:** `uv run python ml/scripts/benchmark_pose_extraction.py` before/after on same video.

---

#### G2. Batch 3D Lifter Windows

**Description:** `ONNXPoseExtractor.estimate_3d` (onnx_extractor.py:58) processes sliding windows sequentially in a while loop (line 79-87). Each window calls `_infer_window` which runs a single ONNX inference (line 111). Windows are independent and can be batched into a single `(num_windows, 81, 17, 3)` tensor.

**Agents:** 1, 5

**Impact:** 3-5x on 3D lifting stage. For a 500-frame video with stride=40, there are ~12 windows. Currently 12 sequential ONNX calls; batched = 1 call.

**Complexity:** Low-Medium. The sliding window overlap averaging (line 89-91) needs adjustment for batched output.

**Dependencies:** None.

**Files:**
- `ml/skating_ml/pose_3d/onnx_extractor.py` (lines 58-92)

---

#### G3. CUDA Graphs for 3D Lifter

**Description:** The 3D lifter (MotionAGFormer ONNX) has fixed input shape `(1, 81, 17, 3)`. CUDA graphs capture the entire GPU computation graph and replay it, eliminating kernel launch overhead (~10us per launch). For the batched windows from G2, this compounds.

**Agents:** 1

**Impact:** 5-15% latency reduction per inference call. Small per-call, but compounds across 12+ windows.

**Complexity:** Low. Single-line change: `session.run()` -> `CUDAExecutionProvider` with `graph_mode=True` in session options, or use `ort.CUDAProviderOptions`.

**Dependencies:** G2 (batching) for maximum benefit.

**Files:**
- `ml/skating_ml/pose_3d/onnx_extractor.py` (line 52: session creation)

---

#### G4. IO Bindings (OrtValue)

**Description:** Currently, each ONNX inference call copies input from CPU numpy array to GPU and output back. Using `OrtValue` with pre-allocated GPU buffers eliminates these copies by keeping data on GPU between calls.

**Agents:** 1

**Impact:** Reduces H2D/D2H transfer overhead. Measurable for small models; for RTMO with 640x640 input, the transfer time is ~0.1ms per frame.

**Complexity:** Medium. Requires managing OrtValue lifetimes and binding patterns.

**Dependencies:** G1 (direct ONNX access).

**Files:**
- `ml/skating_ml/pose_3d/onnx_extractor.py` (line 111)
- New batch RTMO code (from G1)

---

#### G5. Parallel Extras Models

**Description:** In the render loop (web_helpers.py:338-374), per-frame ML inference (depth, flow, segmentation, matting, foot tracking) runs sequentially. These models are independent per frame (except optical flow, which depends on previous frame). Running them via `ThreadPoolExecutor` with separate ONNX sessions would overlap GPU inference.

**Agents:** 1, 5

**Impact:** 1.5-2x when 2+ models are active. When all extras are disabled (default), no impact.

**Complexity:** Medium. Each model needs its own ONNX session. Thread safety of ONNX sessions must be verified.

**Dependencies:** None (only relevant when extras are enabled).

**Files:**
- `ml/skating_ml/web_helpers.py` (lines 338-374: per-frame ML inference)

---

#### G6. ONNX Runtime Thread Configuration

**Description:** No explicit thread configuration is set for ONNX sessions. For GPU-bound workloads (our case), `intra_op=1` prevents CPU thread contention, while `inter_op=2` allows pipelining.

**Agents:** 1, 2

**Impact:** 5-10% for GPU inference. Prevents ONNX from spawning 4+ CPU threads that compete with the main thread.

**Complexity:** Low. Add `SessionOptions.intra_op_num_threads = 1` and `inter_op_num_threads = 2` at session creation.

**Dependencies:** None.

**Files:**
- `ml/skating_ml/pose_3d/onnx_extractor.py` (line 52)
- `ml/skating_ml/pose_estimation/pose_extractor.py` (via rtmlib, may need monkeypatch)
- `ml/skating_ml/extras/` (all model files)

---

#### G7. cuDNN Exhaustive Search

**Description:** ONNX Runtime's CUDA execution provider supports cuDNN benchmark mode, which tries multiple convolution algorithms and picks the fastest for the given input shape.

**Agents:** 1

**Impact:** 5-10% faster inference for convolution-heavy models (RTMO, MotionAGFormer). Adds 1-2s to first inference.

**Complexity:** Low. Set environment variable `ORT_CUDA_CUDNN_CONV_ALGO_SEARCH = 1` (EXHAUSTIVE).

**Dependencies:** None.

**Files:**
- `ml/skating_ml/device.py` (add env var during device init)

---

#### G8. TensorRT Conversion (Deferred)

**Description:** Convert RTMO ONNX model to TensorRT for 1.5-3x inference speedup. TensorRT optimizes the computation graph, fuses layers, and uses FP16 precision.

**Agents:** 1

**Impact:** 1.5-3x RTMO throughput. Highest single-model GPU optimization available.

**Complexity:** High. Requires TensorRT installation (not in Docker image), model conversion, validation, and fallback to ONNX. Risk of numerical differences.

**Dependencies:** G1 (direct ONNX access needed before TensorRT conversion).

**Files:**
- `ml/gpu_server/Containerfile` (add TensorRT)
- New conversion script

**Decision:** Deferred. High complexity and Docker image size increase (TensorRT adds ~1GB). Revisit after G1 is implemented and benchmarked.

---

#### G9. GPU Server Warmup

**Description:** Vast.ai Serverless GPU has cold-start latency. The GPU server (gpu_server/server.py) loads models lazily on first `/process` request. Pre-warming the ONNX sessions during container startup eliminates cold-start latency for subsequent requests.

**Agents:** 1, 3

**Impact:** Eliminates 2-5s cold-start on first request per worker instance.

**Complexity:** Low. Add model loading to FastAPI `@app.on_event("startup")`.

**Dependencies:** None.

**Files:**
- `ml/gpu_server/server.py` (add startup handler)

---

### 3.2 I/O Pipeline (8 opportunities)

#### I1. boto3 -> aiobotocore (gpu_server/server.py)

**Description:** ALL R2 operations in `gpu_server/server.py` are synchronous `boto3` calls inside an `async def process()` handler (server.py:64). `s3.download_file` (line 75), `s3.upload_file` (lines 107, 112, 117) block the entire asyncio event loop for the duration of the network transfer. Since this is a FastAPI server handling concurrent requests, one slow upload blocks all other requests.

**Agents:** 2, 3

**Impact:** Frees the event loop during R2 transfers. For a 50MB video upload at 10MB/s, this unblocks 5s of event loop time.

**Complexity:** Low. Replace `boto3.client("s3")` with `aiobotocore.create_aio_client("s3")`. Use `await client.download_file()` and `await client.upload_file()`.

**Dependencies:** Add `aiobotocore` to `ml/pyproject.toml`.

**Files:**
- `ml/gpu_server/server.py` (lines 14, 53-60, 75, 107, 112, 117)
- `ml/pyproject.toml` (add aiobotocore)

---

#### I2. boto3 -> aiobotocore (backend/app/storage.py)

**Description:** The `storage.py` module creates a new `boto3` client per operation (line 16-25: `_client()` is called in every function). These are used from `worker.py` via `download_file()` (worker.py:261) and `upload_file()` in other paths. The `_client()` function creates a new TCP connection per call.

**Agents:** 2, 3

**Impact:** Eliminates TCP connection setup per operation (~50ms per call). Enables connection reuse.

**Complexity:** Medium. `storage.py` is synchronous and used from both sync and async contexts. The cleanest approach: add async variants (`upload_file_async`, `download_file_async`) alongside existing sync functions.

**Dependencies:** Add `aiobotocore` to `backend/pyproject.toml`.

**Files:**
- `backend/app/storage.py` (lines 16-25, 28-97)
- `ml/skating_ml/worker.py` (lines 261, 365: callers)

---

#### I3. Parallel R2 Uploads in GPU Server

**Description:** After processing, `gpu_server/server.py` uploads 3 independent files sequentially: output video (line 107), poses .npy (line 112), and biomechanics CSV (line 117). These are independent and can run concurrently via `asyncio.gather()`.

**Agents:** 2, 5

**Impact:** Saves 2x upload time (3 serial uploads -> 1 parallel). For a 50MB video + 1MB poses + 100KB CSV, saves ~1s.

**Complexity:** Low. Wrap uploads in `asyncio.gather()`.

**Dependencies:** I1 (aiobotocore).

**Files:**
- `ml/gpu_server/server.py` (lines 105-117)

---

#### I4. Producer-Consumer Frame Buffer

**Description:** Currently, the render loop (web_helpers.py:325-384) decodes and processes frames sequentially: `cap.read()` then `pipe.render_frame()` then `writer.write()`. A background thread could decode frames into a ring buffer while the main thread processes the previous frame.

**Agents:** 1, 2, 5

**Impact:** Overlaps CPU video decode (~2ms/frame) with GPU inference or CPU rendering. For a 500-frame video, saves 0.9-2.25s total.

**Complexity:** Medium. Requires thread-safe queue, buffer management, and careful error handling on video end.

**Dependencies:** None.

**Files:**
- `ml/skating_ml/web_helpers.py` (lines 317-384)
- `ml/skating_ml/visualization/pipeline.py` (render_frame)

---

#### I5. Background Video Encoding

**Description:** `H264Writer.write()` (video_writer.py:42) encodes each frame synchronously via PyAV. This could be decoupled from the render loop by pushing frames to a queue and encoding in a background thread.

**Agents:** 2, 5

**Impact:** Overlaps H.264 encoding (~3ms/frame) with next frame's rendering. Saves 1-2s for 500 frames.

**Complexity:** Medium. Similar to I4 but for the output side. Frame ordering must be preserved.

**Dependencies:** None.

**Files:**
- `ml/skating_ml/utils/video_writer.py` (line 42)
- `ml/skating_ml/web_helpers.py` (line 383: `writer.write(frame)`)

---

#### I6. Async Vast.ai Client

**Description:** `vastai/client.py` uses synchronous `httpx.Client` (lines 39, 94). `process_video_remote()` is called from `worker.py:237` via `asyncio.to_thread()`, which works but occupies a thread pool slot for the entire 600s timeout. Using `httpx.AsyncClient` would free the thread.

**Agents:** 2, 3

**Impact:** Frees thread pool slot during the 600s Vast.ai processing wait. With `max_jobs=1`, this currently doesn't matter, but it matters for higher concurrency (see O3).

**Complexity:** Low. Replace `httpx.post()` with `await httpx.AsyncClient().post()`.

**Dependencies:** None.

**Files:**
- `ml/skating_ml/vastai/client.py` (lines 37-47, 94-99)
- `ml/skating_ml/worker.py` (line 237: remove `asyncio.to_thread` wrapper)

---

#### I7. Redundant Video Opens

**Description:** `pipeline.py:_extract_and_track()` (line 94) opens the video via `extractor.extract_video_tracked()`, then opens it again at line 155 for the first frame (`cv2.VideoCapture(str(video_path))`). The extractor already opens the video internally (pose_extractor.py:219). The first frame could be cached.

**Agents:** 2

**Impact:** Saves one `cv2.VideoCapture` open (~10ms). Minor.

**Complexity:** Low. Return first frame from extraction or cache it.

**Dependencies:** None.

**Files:**
- `ml/skating_ml/pipeline.py` (lines 154-161)
- `ml/skating_ml/pose_estimation/pose_extractor.py` (line 219)

---

#### I8. Valkey Connection Pool

**Description:** `task_manager.py:get_valkey_client()` (line 29) creates a new `aioredis.Redis` connection for every call. Each function (`store_result`, `store_error`, `update_progress`, etc.) calls `get_valkey_client()` when no `valkey` parameter is passed, then closes it in `finally`. For a single task, this means 5+ TCP connections opened and closed.

**Agents:** 2, 3

**Impact:** Saves 5+ TCP connection setups per task (~50ms total). Reduces Valkey connection churn.

**Complexity:** Low. Use a module-level connection pool or pass `valkey` from `worker.py` (which already calls `get_valkey_client()` at line 215).

**Note:** `worker.py` already acquires a valkey client and passes it to `store_result`/`store_error`. But `process_video_task` calls `get_valkey_client()` again at line 215 for status update, then closes at line 331. The issue is when internal code calls task_manager functions without passing the existing client.

**Dependencies:** None.

**Files:**
- `backend/app/task_manager.py` (lines 29-37)
- `ml/skating_ml/worker.py` (line 215)

---

### 3.3 Task Orchestration (9 opportunities)

#### O1. Fix `detect_video_task` Event Loop Blocking (BUG)

**Description:** `detect_video_task` (worker.py:334) is an `async def` but calls `download_file()` (line 365) and `PoseExtractor.preview_persons()` (line 376) synchronously. `download_file` (storage.py:36) uses boto3 which blocks the event loop. `preview_persons` (pose_extractor.py:650) runs GPU inference without `asyncio.to_thread()`. Compare with `process_video_task` (line 237) which correctly wraps the blocking call in `asyncio.to_thread()`.

**Agents:** 2, 3

**Impact:** Fixes actual bug where the arq event loop is frozen during detection (~5s), preventing any other tasks from being checked or cancelled.

**Complexity:** Low. Wrap blocking calls in `asyncio.to_thread()`.

**Dependencies:** None.

**Files:**
- `ml/skating_ml/worker.py` (lines 363-376: wrap in `asyncio.to_thread`)

---

#### O2. Wire `analyze_async()` into GPU Server

**Description:** `AnalysisPipeline.analyze_async()` (pipeline.py:605) exists with parallel stage execution (3D lifting || phase detection, metrics || reference loading), but it is never called. The GPU server's `process_video_pipeline` (web_helpers.py:136) runs analysis sequentially after the render loop (lines 404-423). Additionally, the analysis could run in parallel with rendering.

**Agents:** 3, 5

**Impact:** Parallel analysis stages save ~100ms (3D lift overlaps with phase detection). Analysis overlapping with render saves the full analysis time (~200ms) from the critical path.

**Complexity:** Medium. Requires threading the async pipeline into the sync render loop, or splitting into separate tasks.

**Dependencies:** O1 (event loop must not be blocked for async to work).

**Files:**
- `ml/skating_ml/web_helpers.py` (lines 404-423)
- `ml/skating_ml/pipeline.py` (lines 605-751: analyze_async)

---

#### O3. Increase `max_jobs` for Vast.ai Path

**Description:** `WorkerSettings.max_jobs` (worker.py:455) defaults to 1 (config.py:116). When dispatching to Vast.ai, the worker is just a dispatcher -- it doesn't use GPU locally. Processing 5+ jobs concurrently would improve throughput.

**Agents:** 3

**Impact:** 5x job throughput for Vast.ai path. Each job is just HTTP calls + R2 transfers; no GPU contention.

**Complexity:** Low. Change config default or make it path-dependent.

**Dependencies:** I6 (async Vast.ai client to avoid thread pool exhaustion). I1/I2 (async R2 to avoid event loop blocking).

**Files:**
- `backend/app/config.py` (line 116: `worker_max_jobs`)
- `ml/skating_ml/worker.py` (line 455)

---

#### O4. SSE Streaming for Progress

**Description:** The frontend currently polls task status via `GET /process/{task_id}/status`. `sse-starlette` is already in dependencies. Server-Sent Events would provide real-time progress updates without polling overhead.

**Agents:** 3

**Impact:** Real-time progress for users. Reduces polling load on backend.

**Complexity:** Medium. Add SSE endpoint, have worker publish progress to Valkey pub/sub, SSE endpoint subscribes.

**Dependencies:** None.

**Files:**
- `backend/app/routes/process.py` (add SSE endpoint)
- `ml/skating_ml/worker.py` (publish progress to pub/sub)
- `backend/app/task_manager.py` (add pub/sub helpers)

---

#### O5. Cancellation Checks

**Description:** The worker never calls `is_cancelled()` (task_manager.py:176). If a user cancels a processing job, the GPU continues working until completion, wasting Vast.ai GPU minutes.

**Agents:** 3

**Impact:** Saves GPU minutes on cancelled jobs. Currently wastes full processing time per cancellation.

**Complexity:** Low. Add `is_cancelled()` check in the render loop (web_helpers.py:326 already checks `cancel_event`, but this is only for local processing, not Vast.ai).

**Dependencies:** None.

**Files:**
- `ml/skating_ml/worker.py` (add cancellation check before/after Vast.ai dispatch)
- `ml/gpu_server/server.py` (add cancellation parameter to ProcessRequest)

---

#### O6. Parallel Post-Processing in Worker

**Description:** After Vast.ai returns results, `process_video_task` runs `_sample_poses()`, `_compute_frame_metrics()`, and `store_result()` sequentially (worker.py:268-288). These are independent: sampling and metrics computation operate on the same poses array but produce different outputs.

**Agents:** 3, 5

**Impact:** Saves ~50ms (metrics computation time) from the post-processing phase.

**Complexity:** Low. Use `asyncio.gather()` with `asyncio.to_thread()` wrappers.

**Dependencies:** None.

**Files:**
- `ml/skating_ml/worker.py` (lines 268-288)

---

#### O7. Vast.ai Worker URL Caching

**Description:** `_get_worker_url()` (vastai/client.py:37) calls Vast.ai's `/route` endpoint every time. If multiple videos are processed in quick succession, the same worker URL could be reused for 60s (TTL).

**Agents:** 3

**Impact:** Saves 1-3s per request (HTTP round-trip to Vast.ai route). Avoids unnecessary cold starts.

**Complexity:** Low. Add a TTL cache (e.g., `functools.lru_cache` with TTL, or simple dict with timestamp).

**Dependencies:** None.

**Files:**
- `ml/skating_ml/vastai/client.py` (lines 37-47)

---

#### O8. Per-Request arq Pool Creation

**Description:** Every call to enqueue an arq job (from backend routes) creates a new Redis/Valkey connection. The enqueue functions in the backend routes should share a connection pool.

**Agents:** 3

**Impact:** Reduces connection overhead per job submission.

**Complexity:** Low. Use a shared connection pool in the FastAPI app lifespan.

**Dependencies:** None.

**Files:**
- `backend/app/routes/process.py` (enqueue call)
- `backend/app/routes/detect.py` (enqueue call)

---

#### O9. Job Priority Queues

**Description:** `detect_video_task` (fast, ~5s) and `process_video_task` (slow, ~60-300s) share the same queue (`skating:queue`). A slow process job can block detection jobs. Separate queues with different priorities would ensure detection remains responsive.

**Agents:** 3

**Impact:** Detection jobs complete in ~5s regardless of processing queue depth.

**Complexity:** Medium. Requires separate arq worker processes for each queue, or arq's built-in job prioritization.

**Dependencies:** O3 (increased max_jobs makes this more important).

**Files:**
- `ml/skating_ml/worker.py` (line 454: queue_name)
- `backend/app/config.py` (add queue configuration)

---

### 3.4 CPU Preprocessing (6 opportunities)

#### C1. Vectorize `_compute_frame_metrics`

**Description:** `_compute_frame_metrics()` (worker.py:59) is a pure Python loop iterating over every frame (line 75: `for pose in poses`). It computes knee angles, hip angles, trunk lean, and CoM height using per-element numpy operations inside a Python loop. This should be vectorized to operate on the full `(N, 17, 3)` array at once.

**Agents:** 4

**Impact:** Eliminates Python loop overhead for ~500 frames. The function is called once per processed video. Estimated 10-50x speedup for this specific function.

**Complexity:** Low-Medium. Requires converting per-frame angle calculations to batch operations using numpy broadcasting.

**Dependencies:** None.

**Files:**
- `ml/skating_ml/worker.py` (lines 59-180)

---

#### C2. Vectorize `calculate_com_trajectory` and `calculate_center_of_mass`

**Description:** Physics engine functions iterate per-frame when computing CoM trajectories. These operate on `(N, 17, 3)` arrays and can be vectorized using numpy operations across the frame dimension.

**Agents:** 4

**Impact:** Eliminates Python loop overhead. Functions already use numpy internally but may have per-frame Python loops.

**Complexity:** Low.

**Dependencies:** None.

**Files:**
- `ml/skating_ml/analysis/physics_engine.py`

---

#### C3. Vectorize `calculate_moment_of_inertia`

**Description:** Similar to C2, this function may have per-frame Python loops that can be vectorized.

**Agents:** 4

**Impact:** Eliminates Python loop overhead.

**Complexity:** Low.

**Dependencies:** None.

**Files:**
- `ml/skating_ml/analysis/physics_engine.py`

---

#### C4. Vectorize `PoseNormalizer.normalize` and `normalize_poses`

**Description:** Pose normalization iterates per-frame for root-centering and scale normalization. These are embarrassingly parallel across frames.

**Agents:** 4

**Impact:** Small -- normalization is already fast (<10ms). But eliminates any remaining Python loops.

**Complexity:** Low.

**Dependencies:** None.

**Files:**
- `ml/skating_ml/pose_estimation/normalizer.py`

---

#### C5. Parallel DTW + Physics

**Description:** In `analyze_async()`, DTW alignment (pipeline.py:694-700) and physics calculations (pipeline.py:702-724) are independent when both a reference and 3D poses are available. They could run in parallel via `asyncio.gather()`.

**Agents:** 4, 5

**Impact:** Saves ~50ms (physics computation time) when both are active.

**Complexity:** Low. Already partially parallelized in `analyze_async()`.

**Dependencies:** O2 (wire analyze_async).

**Files:**
- `ml/skating_ml/pipeline.py` (lines 694-724)

---

#### C6. OMP_NUM_THREADS Configuration

**Description:** When using `ProcessPoolExecutor` alongside ONNX Runtime, both may try to use all CPU cores, causing over-subscription. Setting `OMP_NUM_THREADS` appropriately prevents this.

**Agents:** 4

**Impact:** Prevents CPU thrashing when multiple parallel paths are active.

**Complexity:** Low. Set environment variable.

**Dependencies:** Only relevant if multiprocessing is used (currently not).

**Files:**
- `ml/skating_ml/worker.py` (set env var at module level)

---

### 3.5 Pipeline DAG (5 opportunities)

#### D1. Render + Analysis Parallelism

**Description:** The render loop (web_helpers.py:325-384) and biomechanics analysis (web_helpers.py:404-423) are completely independent. Render needs only `PreparedPoses`; analysis needs `poses_norm + fps + element_type`. They can run concurrently via `ThreadPoolExecutor`: render in one thread, analysis in another.

**Agents:** 4, 5

**Impact:** Saves the full analysis time (~200ms) from the critical path. The render loop finishes sooner because it doesn't wait for analysis.

**Complexity:** Low-Medium. Start analysis thread before render loop, join after render completes.

**Dependencies:** None.

**Files:**
- `ml/skating_ml/web_helpers.py` (lines 316-423)

---

#### D2. Early Termination Paths

**Description:** When CorrectiveLens is disabled (default), the code still resolves the model path (pipeline.py:419: `Path("data/models/motionagformer-s-ap3d.onnx").exists()`). When no element_type is provided, the entire analysis block (lines 270-342) is skipped but normalization and smoothing still run. These are minor but measurable.

**Agents:** 5

**Impact:** Saves ~5-10ms per skipped model path resolution. Small.

**Complexity:** Low. Add early returns or guard clauses.

**Dependencies:** None.

**Files:**
- `ml/skating_ml/pipeline.py` (lines 238-266: 3D lift block)

---

#### D3. Parallel Comparison Extraction

**Description:** The comparison tool (visualization/comparison.py) extracts poses for both the athlete video and reference video. These are independent and can run in parallel.

**Agents:** 4, 5

**Impact:** 2x pose extraction for comparison workflow.

**Complexity:** Low. Use `ThreadPoolExecutor` for two `prepare_poses()` calls.

**Dependencies:** None.

**Files:**
- `ml/skating_ml/visualization/comparison.py`

---

#### D4. Phase-Aware Smoothing Parallelism

**Description:** When using `smooth_phase_aware()`, independent phase segments (pre-takeoff, flight, post-landing) are smoothed sequentially. These segments are independent and could be smoothed in parallel.

**Agents:** 4

**Impact:** Saves ~50ms for phase-aware smoothing. Small.

**Complexity:** Low.

**Dependencies:** None.

**Files:**
- `ml/skating_ml/utils/smoothing.py`

---

#### D5. Streaming Pipeline (Deferred)

**Description:** Process the video in chunks instead of loading all frames. This reduces peak memory but does NOT reduce wall time (same total work, more overhead for chunking).

**Agents:** 1, 5

**Impact:** Reduces peak memory from ~68MB (500 frames * 17kp * 3 * 4 bytes) to ~7MB per chunk. No time savings.

**Complexity:** High. Requires refactoring the entire pipeline to operate on chunks. Tracking across chunks is complex.

**Decision:** Deferred. Memory is not a bottleneck on 4GB VRAM / 16GB RAM systems.

---

---

## 4. Conflicts & Trade-offs

### 4.1 Batch Inference vs. VRAM Limits (4GB)

**Conflict:** G1 (batch RTMO) and G2 (batch 3D lifter) increase peak VRAM usage by processing multiple frames simultaneously. On RTX 3050 Ti (4GB VRAM), this may cause OOM.

**Resolution:** RTMO input is `(B, 3, 640, 640)` at FP32 = B * 4.7MB. For batch_size=8, that's 37.6MB. The model itself is ~50MB. Total ~90MB, well within 4GB. For 3D lifter, batch_size=12 windows of `(12, 81, 17, 3)` = 1.2MB. No VRAM concern.

**Action:** Implement G1 and G2 with configurable batch sizes. Default batch_size=8 for RTMO, batch_size=12 for 3D lifter. Add VRAM monitoring.

### 4.2 Batch Inference vs. Tracking Accuracy

**Conflict:** G1 (batch RTMO) processes multiple frames at once, but tracking (Sports2D/DeepSORT) requires per-frame detection results to maintain temporal consistency. Batch inference produces results for frames that may be processed out of order.

**Resolution:** Batch inference outputs are in order (batch preserves frame sequence). Tracking still operates per-frame on batch results. No conflict.

**Action:** Process batch -> feed results sequentially to tracker. The batch only affects ONNX inference, not tracking logic.

### 4.3 Parallel Extras vs. VRAM

**Conflict:** G5 (parallel extras models) loads depth (115MB), SAM2 (180MB), optical flow (50MB), matting (20MB), foot tracker (15MB), and LAMA (240MB) simultaneously. Total: ~620MB model VRAM + input tensors. Combined with RTMO (~50MB), this approaches 4GB.

**Resolution:** Extras are optional and rarely all enabled simultaneously. The `ModelRegistry` already manages VRAM budgets. When multiple models exceed VRAM, it falls back to CPU or skips models.

**Action:** Add explicit VRAM budget check before enabling parallel extras. Default: disable parallel extras if total model VRAM > 2GB.

### 4.4 Async R2 vs. Backward Compatibility

**Conflict:** I1/I2 (boto3->aiobotocore) changes the storage API from sync to async. All callers must be updated.

**Resolution:** Add async variants alongside existing sync functions. New code uses async; old code continues to work.

**Action:** `storage.py` gets `upload_file_async()`, `download_file_async()` etc. `gpu_server/server.py` (async context) migrates immediately. `worker.py` (mixed sync/async) migrates incrementally.

### 4.5 `max_jobs > 1` vs. Single GPU

**Conflict:** O3 (increase max_jobs) allows concurrent tasks, but local GPU path shares a single RTX 3050 Ti. Multiple jobs would compete for GPU memory.

**Resolution:** Only increase `max_jobs` for Vast.ai path (no local GPU). For local GPU path, keep `max_jobs=1`. Add a config flag: `worker_max_jobs_local=1`, `worker_max_jobs_remote=5`.

**Action:** Make `max_jobs` path-dependent in `WorkerSettings`.

### 4.6 Vectorization vs. Readability

**Conflict:** C1-C4 (vectorize per-frame loops) make code less readable by replacing clear Python loops with numpy broadcasting operations.

**Resolution:** Add helper functions with clear names (e.g., `compute_knee_angles_batch()`). Keep original per-frame versions as fallbacks for debugging. Add type hints and docstrings.

**Action:** Create vectorized versions in new functions, deprecate old versions after validation.

---

## 5. Implementation Phases

### Phase 1: Quick Wins (1-2 days each)

**Goal:** Fix bugs, eliminate event loop blocking, add low-risk optimizations.

| ID | Task | Effort | Files |
|----|------|--------|-------|
| O1 | Fix `detect_video_task` event loop blocking | 0.5d | `worker.py` |
| G6 | ONNX Runtime thread configuration | 0.5d | `onnx_extractor.py`, `device.py` |
| G7 | cuDNN exhaustive search env var | 0.5d | `device.py` |
| I8 | Valkey connection reuse in worker | 0.5d | `task_manager.py`, `worker.py` |
| O7 | Vast.ai worker URL caching (60s TTL) | 0.5d | `vastai/client.py` |
| G9 | GPU server warmup on startup | 0.5d | `gpu_server/server.py` |
| O6 | Parallel post-processing in worker | 0.5d | `worker.py` |
| D2 | Early termination for disabled features | 0.5d | `pipeline.py` |
| C6 | OMP_NUM_THREADS configuration | 0.5d | `worker.py` |
| I7 | Cache first frame from extraction | 0.5d | `pipeline.py`, `pose_extractor.py` |

**Phase 1 Validation:** Run `uv run python ml/scripts/benchmark_pose_extraction.py` before and after. Compare `PipelineProfiler` output.

---

### Phase 2: Medium Effort (3-5 days each)

**Goal:** I/O async migration, vectorization, pipeline parallelism.

| ID | Task | Effort | Files |
|----|------|--------|-------|
| I1 | boto3 -> aiobotocore in GPU server | 2d | `gpu_server/server.py`, `pyproject.toml` |
| I2 | Add async storage variants in backend | 2d | `backend/app/storage.py`, `backend/pyproject.toml` |
| I3 | Parallel R2 uploads (depends on I1) | 1d | `gpu_server/server.py` |
| I6 | Async Vast.ai client | 1d | `vastai/client.py`, `worker.py` |
| C1 | Vectorize `_compute_frame_metrics` | 2d | `worker.py` |
| C2 | Vectorize physics engine functions | 2d | `physics_engine.py` |
| D1 | Render + analysis parallelism | 2d | `web_helpers.py` |
| O3 | Path-dependent `max_jobs` (depends on I1, I6) | 1d | `config.py`, `worker.py` |
| G2 | Batch 3D lifter windows | 2d | `onnx_extractor.py` |
| G5 | Parallel extras models | 3d | `web_helpers.py`, `extras/` |

**Phase 2 Validation:** Benchmark full pipeline end-to-end before and after each change. Use `PipelineProfiler` for per-stage timing.

---

### Phase 3: Architectural (1-2 weeks each)

**Goal:** Direct ONNX batch inference, SSE streaming, job priority.

| ID | Task | Effort | Files |
|----|------|--------|-------|
| G1 | Direct ONNX batch RTMO inference | 10d | `pose_extractor.py`, new `batch_rtmo.py` |
| O2 | Wire `analyze_async()` into GPU server | 5d | `web_helpers.py`, `pipeline.py` |
| O4 | SSE streaming for progress | 5d | `routes/process.py`, `worker.py`, `task_manager.py` |
| O9 | Job priority queues | 5d | `worker.py`, `config.py`, new worker process |
| I4 | Producer-consumer frame buffer | 5d | `web_helpers.py`, new `frame_buffer.py` |
| I5 | Background video encoding | 3d | `video_writer.py`, `web_helpers.py` |

**Phase 3 Validation:** Full benchmark suite with representative videos (short 5s, medium 15s, long 60s). Compare against Phase 2 baseline.

---

## 6. Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Batch RTMO breaks tracking | Medium | High | Unit tests on tracking accuracy. A/B comparison on test videos. Fallback to per-frame. |
| VRAM OOM with batch inference | Low | Medium | Configurable batch_size. VRAM monitoring. Fallback to batch_size=1. |
| aiobotocore incompatibility | Low | Medium | Keep sync boto3 as fallback. Test with actual R2 endpoints. |
| CUDA graph capture fails | Low | Low | Graph capture is optional; falls back to normal inference. |
| Vectorization introduces numerical drift | Low | Medium | Compare vectorized output with per-frame output. Tolerance: 1e-6. |
| `max_jobs > 1` causes resource contention | Medium | Medium | Only apply to Vast.ai path. Monitor GPU/CPU/memory. |
| SSE connection drops | Medium | Low | Client-side reconnect with last-event-id. Fallback to polling. |
| TensorRT model incompatibility | High | High | Deferred to post-G1. Requires extensive validation. |
| Thread safety in ONNX sessions | Low | High | ONNX Runtime sessions are thread-safe for run(). Each thread needs its own session. |
| Pipeline changes break existing tests | Medium | Medium | Run full test suite after each change. Add benchmark regression tests. |

---

## 7. Metrics & Validation

### 7.1 Benchmark Suite

All optimizations must be validated with before/after benchmarks. The existing `PipelineProfiler` (utils/profiling.py) provides per-stage timing.

**Benchmark script:** `uv run python ml/scripts/benchmark_pose_extraction.py`

**Test videos:**
- Short: 5s (~150 frames at 30fps)
- Medium: 15s (~450 frames at 30fps) -- primary benchmark
- Long: 60s (~1800 frames at 30fps)

### 7.2 Key Metrics

| Metric | How to Measure | Target |
|--------|---------------|--------|
| Total pipeline wall time | `PipelineProfiler.total_wall_time_s` | Measure before/after per phase |
| RTMO inference throughput | `profiler.record("rtmo_inference_loop", ...)` | Frames/second |
| 3D lifting throughput | `profiler.record("3d_lift_and_blade", ...)` | Windows/second |
| Event loop block time | `asyncio` debug mode logging | 0ms blocking in async functions |
| VRAM peak usage | `pynvml` before/after each stage | < 3.5GB (leave 500MB headroom) |
| R2 transfer time | Time `download_file`/`upload_file` calls | Measure before/after |
| Worker job throughput | Jobs completed per minute | Measure with `max_jobs=1` vs `5` |

### 7.3 Regression Tests

**Tracking accuracy:** Run `uv run python -m pytest ml/tests/tracking/` after any pose extraction changes. Compare tracked person consistency.

**Numerical accuracy:** For vectorized functions (C1-C4), compare output with original per-frame implementation. Tolerance: `np.allclose(original, vectorized, atol=1e-6)`.

**Pipeline output:** Full pipeline output (poses, metrics, recommendations) must be identical before and after optimization changes. Hash comparison of output .npy files.

### 7.4 Per-Phase Validation Checkpoints

**After Phase 1:** Event loop no longer blocks. All existing tests pass. Benchmark shows no regression.

**After Phase 2:** R2 operations are async. `_compute_frame_metrics` is vectorized. Render+analysis overlap is active. Benchmark shows measurable improvement.

**After Phase 3:** Batch RTMO is active. SSE streaming works. Job priority is functional. Benchmark shows significant improvement.

---

## 8. Appendix: File Reference Map

| File | Role | Key Lines |
|------|------|-----------|
| `ml/skating_ml/worker.py` | arq worker | L191: process_video_task, L334: detect_video_task, L451: WorkerSettings |
| `ml/skating_ml/pipeline.py` | Analysis orchestrator | L94: _extract_and_track, L176: analyze, L605: analyze_async |
| `ml/skating_ml/web_helpers.py` | GPU server pipeline | L136: process_video_pipeline, L325: render loop |
| `ml/skating_ml/pose_estimation/pose_extractor.py` | RTMO extraction | L48: PoseExtractor, L141: extract_video_tracked |
| `ml/skating_ml/pose_estimation/batch_extractor.py` | Batch RTMO (no-op) | L44: BatchPoseExtractor, L276: per-frame loop |
| `ml/skating_ml/pose_3d/onnx_extractor.py` | 3D lifter | L58: estimate_3d, L94: _infer_window |
| `ml/gpu_server/server.py` | Vast.ai GPU server | L64: process, L75: sync download, L107-117: sync uploads |
| `ml/skating_ml/vastai/client.py` | Vast.ai HTTP client | L37: _get_worker_url, L50: process_video_remote |
| `ml/skating_ml/device.py` | GPU/CPU config | L88: DeviceConfig, L134: onnx_providers |
| `ml/skating_ml/utils/profiling.py` | Pipeline profiler | L33: PipelineProfiler, L72: record |
| `ml/skating_ml/utils/video_writer.py` | H264 output | L21: H264Writer, L42: write |
| `backend/app/storage.py` | R2/S3 operations | L16: _client, L28: upload_file, L36: download_file |
| `backend/app/task_manager.py` | Valkey task state | L29: get_valkey_client, L90: store_result |
| `backend/app/config.py` | App settings | L116: worker_max_jobs |
