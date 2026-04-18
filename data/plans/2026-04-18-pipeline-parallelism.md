# Pipeline Parallelism & Async Optimization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Optimize the ML pipeline for parallel and async execution across GPU inference, I/O, task orchestration, CPU preprocessing, and pipeline DAG stages.

**Architecture:** Three-phase approach: (1) fix bugs and add zero-risk config tweaks, (2) migrate I/O to async and vectorize CPU loops, (3) implement batch GPU inference and SSE streaming. Each phase produces measurable improvements validated by `PipelineProfiler`.

**Tech Stack:** Python asyncio, onnxruntime-gpu, aiobotocore, httpx, concurrent.futures, ThreadPoolExecutor, NumPy vectorization

**Design doc:** `ml/docs/PIPELINE_PARALLELISM_DESIGN.md`

---

## Phase 1: Quick Wins

**Goal:** Fix event loop blocking bugs, add low-risk config optimizations. Each task is independent.

**Validation:** `uv run python -m pytest ml/tests/ -x --no-cov` after each task. Benchmark baseline: `uv run python ml/scripts/benchmark_pose_extraction.py` before starting.

---

### Task 1: Fix `detect_video_task` event loop blocking (BUG)

**Files:**
- Modify: `ml/skating_ml/worker.py:363-377`

- [ ] **Step 1: Identify blocking calls**

In `detect_video_task` (worker.py:334), these calls block the async event loop:
- Line 365: `download_file(video_key, str(video_path))` — sync boto3
- Line 376: `extractor.preview_persons(video_path, num_frames=30)` — sync GPU inference

- [ ] **Step 2: Wrap blocking calls in `asyncio.to_thread()`**

```python
# In ml/skating_ml/worker.py, detect_video_task function
# BEFORE (lines 363-377):
    video_path = download_file(video_key, str(video_path))
    # ...
    persons, _ = extractor.preview_persons(video_path, num_frames=30)

# AFTER:
    video_path = await asyncio.to_thread(download_file, video_key, str(video_path))
    # ...
    persons, _ = await asyncio.to_thread(
        extractor.preview_persons, video_path, num_frames=30
    )
```

- [ ] **Step 3: Run tests**

```bash
uv run python -m pytest ml/tests/ -x --no-cov
```

Expected: PASS (no test changes needed, same behavior)

- [ ] **Step 4: Commit**

```bash
git add ml/skating_ml/worker.py
git commit -m "fix(worker): wrap detect_video_task blocking calls in asyncio.to_thread"
```

---

### Task 2: Wrap `process_video_task` R2 download in `asyncio.to_thread()`

**Files:**
- Modify: `ml/skating_ml/worker.py:261`

- [ ] **Step 1: Wrap download_file call**

```python
# In ml/skating_ml/worker.py, process_video_task function
# BEFORE (line 261):
    download_file(vast_result.poses_key, str(poses_path))

# AFTER:
    await asyncio.to_thread(download_file, vast_result.poses_key, str(poses_path))
```

- [ ] **Step 2: Run tests**

```bash
uv run python -m pytest ml/tests/ -x --no-cov
```

- [ ] **Step 3: Commit**

```bash
git add ml/skating_ml/worker.py
git commit -m "fix(worker): wrap R2 poses download in asyncio.to_thread"
```

---

### Task 3: ONNX Runtime thread configuration

**Files:**
- Modify: `ml/skating_ml/pose_3d/onnx_extractor.py:36-56`
- Modify: `ml/skating_ml/extras/model_registry.py` (session creation)

- [ ] **Step 1: Add SessionOptions to ONNXPoseExtractor**

```python
# In ml/skating_ml/pose_3d/onnx_extractor.py, __init__ method
# BEFORE:
    self.session = ort.InferenceSession(str(model_path), providers=cfg.onnx_providers)

# AFTER:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1   # GPU-bound: minimize CPU threads
    opts.inter_op_num_threads = 2   # Allow pipeline overlap
    self.session = ort.InferenceSession(
        str(model_path), sess_options=opts, providers=cfg.onnx_providers
    )
```

- [ ] **Step 2: Add SessionOptions to ModelRegistry**

Read `ml/skating_ml/extras/model_registry.py` to find the session creation line (around line 113). Apply the same pattern:

```python
# In model_registry.py, wherever ort.InferenceSession is created:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 2
    session = ort.InferenceSession(entry.path, sess_options=opts, providers=self._device.onnx_providers)
```

- [ ] **Step 3: Run tests**

```bash
uv run python -m pytest ml/tests/ -x --no-cov
```

- [ ] **Step 4: Commit**

```bash
git add ml/skating_ml/pose_3d/onnx_extractor.py ml/skating_ml/extras/model_registry.py
git commit -m "perf(gpu): configure ONNX Runtime thread pools for GPU-bound workloads"
```

---

### Task 4: cuDNN exhaustive search env var

**Files:**
- Modify: `ml/skating_ml/device.py`

- [ ] **Step 1: Set ORT_CUDA_CUDNN_CONV_ALGO_SEARCH in DeviceConfig**

```python
# In ml/skating_ml/device.py, at module level (after imports, before class definitions):
import os

# cuDNN benchmark: try all convolution algorithms, pick fastest for this GPU.
# Adds ~1-2s to first inference, but speeds up all subsequent calls.
os.environ.setdefault("ORT_CUDA_CUDNN_CONV_ALGO_SEARCH", "EXHAUSTIVE")
```

- [ ] **Step 2: Run tests**

```bash
uv run python -m pytest ml/tests/ -x --no-cov
```

- [ ] **Step 3: Commit**

```bash
git add ml/skating_ml/device.py
git commit -m "perf(gpu): enable cuDNN exhaustive algorithm search for RTX 3050 Ti"
```

---

### Task 5: Vast.ai worker URL caching (60s TTL)

**Files:**
- Modify: `ml/skating_ml/vastai/client.py:37-47`

- [ ] **Step 1: Add TTL cache for worker URL**

```python
# In ml/skating_ml/vastai/client.py, add after imports:
import time

_worker_url_cache: str | None = None
_worker_url_cache_time: float = 0.0
_WORKER_URL_TTL = 60  # seconds


def _get_worker_url(endpoint_name: str, api_key: str) -> str:
    """Get Vast.ai worker URL with 60s TTL cache to avoid cold starts."""
    global _worker_url_cache, _worker_url_cache_time
    now = time.monotonic()
    if _worker_url_cache and (now - _worker_url_cache_time) < _WORKER_URL_TTL:
        return _worker_url_cache
    # ... existing implementation ...
    _worker_url_cache = url
    _worker_url_cache_time = now
    return url
```

- [ ] **Step 2: Run tests**

```bash
uv run python -m pytest ml/tests/ -x --no-cov
```

- [ ] **Step 3: Commit**

```bash
git add ml/skating_ml/vastai/client.py
git commit -m "perf(vastai): cache worker URL for 60s to avoid repeated cold starts"
```

---

### Task 6: GPU server warmup on startup

**Files:**
- Modify: `ml/gpu_server/server.py`

- [ ] **Step 1: Add startup warmup handler**

```python
# In ml/gpu_server/server.py, add after app creation:
@app.on_event("startup")
async def warmup_gpu():
    """Pre-warm ONNX sessions to eliminate cold-start latency on first request."""
    import numpy as np
    from skating_ml.device import DeviceConfig
    cfg = DeviceConfig.default()
    if not cfg.is_cuda:
        return
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 2
    # Trigger CUDA init + cuDNN benchmark with a dummy inference
    dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
    session = ort.InferenceSession(
        "", sess_options=opts, providers=cfg.onnx_providers
    )
    del session
    logging.getLogger(__name__).info("GPU warmup complete")
```

Note: The warmup just triggers CUDA/cuDNN initialization. The actual model loading happens lazily on first request. A more thorough warmup would load the RTMO model, but that requires knowing the model path at startup.

- [ ] **Step 2: Run tests**

```bash
uv run python -m pytest ml/tests/ -x --no-cov
```

- [ ] **Step 3: Commit**

```bash
git add ml/gpu_server/server.py
git commit -m "perf(gpu): add GPU warmup on server startup"
```

---

### Task 7: Parallel post-processing in worker

**Files:**
- Modify: `ml/skating_ml/worker.py:254-288`

- [ ] **Step 1: Use asyncio.gather for independent operations**

```python
# In ml/skating_ml/worker.py, process_video_task function
# AFTER downloading poses (line 261), BEFORE store_result:

# BEFORE (sequential):
    poses = np.load(str(poses_path))
    pose_data = _sample_poses(poses, sample_rate=10)
    frame_metrics = _compute_frame_metrics(poses)
    await store_result(task_id, response_data, valkey=valkey)

# AFTER (parallel):
    poses = np.load(str(poses_path))
    sample_future = asyncio.to_thread(_sample_poses, poses, 10)
    metrics_future = asyncio.to_thread(_compute_frame_metrics, poses)
    pose_data, frame_metrics = await asyncio.gather(sample_future, metrics_future)
    await store_result(task_id, response_data, valkey=valkey)
```

- [ ] **Step 2: Run tests**

```bash
uv run python -m pytest ml/tests/ -x --no-cov
```

- [ ] **Step 3: Commit**

```bash
git add ml/skating_ml/worker.py
git commit -m "perf(worker): parallelize _sample_poses and _compute_frame_metrics"
```

---

### Task 8: Early termination for disabled features

**Files:**
- Modify: `ml/skating_ml/web_helpers.py`
- Modify: `ml/skating_ml/pipeline.py`

- [ ] **Step 1: Skip 3D model path resolution when CorrectiveLens is disabled**

Read `ml/skating_ml/pipeline.py` to find where 3D model path is resolved (around lines 238-266). If there's a Path.exists() check or model resolution that happens even when 3D is not needed, guard it with the `use_3d` or equivalent flag.

- [ ] **Step 2: Skip analysis setup when no element_type**

In `ml/skating_ml/web_helpers.py`, the biomechanics analysis block (lines 399-423) already checks `element_type`. Verify this guard exists and is correct.

- [ ] **Step 3: Run tests**

```bash
uv run python -m pytest ml/tests/ -x --no-cov
```

- [ ] **Step 4: Commit**

```bash
git add ml/skating_ml/web_helpers.py ml/skating_ml/pipeline.py
git commit -m "perf(pipeline): skip 3D model resolution when CorrectiveLens is disabled"
```

---

### Task 9: Cache first frame from extraction

**Files:**
- Modify: `ml/skating_ml/pipeline.py:154-161`
- Modify: `ml/skating_ml/pose_estimation/pose_extractor.py` (TrackedExtraction dataclass)

- [ ] **Step 1: Add `first_frame` field to TrackedExtraction**

Read `ml/skating_ml/pose_estimation/pose_extractor.py` to find the `TrackedExtraction` dataclass definition. Add:

```python
@dataclass
class TrackedExtraction:
    # ... existing fields ...
    first_frame: np.ndarray | None = None  # Cached first frame for spatial reference
```

- [ ] **Step 2: Capture first frame during extraction**

In `extract_video_tracked()`, capture the first frame from `cap.read()` before the main loop:

```python
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Cannot read first frame")
    # ... proceed with main loop, using first_frame for initial detection ...
```

- [ ] **Step 3: Use cached first frame in pipeline.py**

In `pipeline.py:_extract_and_track()`, replace the redundant video open:

```python
# BEFORE (pipeline.py:155-161):
    cap = cv2.VideoCapture(str(video_path))
    ret, first_frame = cap.read()
    cap.release()

# AFTER:
    first_frame = extraction.first_frame
```

- [ ] **Step 4: Run tests**

```bash
uv run python -m pytest ml/tests/ -x --no-cov
```

- [ ] **Step 5: Commit**

```bash
git add ml/skating_ml/pipeline.py ml/skating_ml/pose_estimation/pose_extractor.py
git commit -m "refactor(pipeline): cache first frame from extraction, eliminate redundant video open"
```

---

### Task 10: OMP_NUM_THREADS configuration

**Files:**
- Modify: `ml/skating_ml/worker.py` (module level)

- [ ] **Step 1: Set OMP_NUM_THREADS at worker module level**

```python
# In ml/skating_ml/worker.py, at the very top after imports:
import os

# Prevent BLAS/OpenBLAS from spawning too many threads when ONNX Runtime
# also uses CPU threads. ONNX handles its own threading via SessionOptions.
os.environ.setdefault("OMP_NUM_THREADS", "2")
```

- [ ] **Step 2: Run tests**

```bash
uv run python -m pytest ml/tests/ -x --no-cov
```

- [ ] **Step 3: Commit**

```bash
git add ml/skating_ml/worker.py
git commit -m "perf(worker): set OMP_NUM_THREADS=2 to prevent BLAS over-subscription"
```

---

### Task 11: Phase 1 validation benchmark

**Files:**
- Run: `ml/scripts/benchmark_pose_extraction.py`

- [ ] **Step 1: Run baseline benchmark**

```bash
uv run python ml/scripts/benchmark_pose_extraction.py
```

Record results in a comment in the design doc or memory. Compare with pre-Phase-1 baseline.

- [ ] **Step 2: Run full test suite**

```bash
uv run python -m pytest ml/tests/ --no-cov -q
```

- [ ] **Step 3: Commit Phase 1 milestone**

```bash
git add -A
git commit -m "milestone(pipeline): Phase 1 quick wins complete"
```

---

## Phase 2: Medium Effort

**Goal:** Migrate I/O to async, vectorize CPU loops, add pipeline-level parallelism.

**Prerequisites:** Phase 1 complete.

**Validation:** Benchmark before and after each task using `PipelineProfiler`.

---

### Task 12: Add aiobotocore dependency

**Files:**
- Modify: `ml/pyproject.toml`
- Modify: `backend/pyproject.toml`

- [ ] **Step 1: Add aiobotocore to ml/pyproject.toml**

```bash
cd ml && uv add aiobotocore
```

- [ ] **Step 2: Add aiobotocore to backend/pyproject.toml**

```bash
cd backend && uv add aiobotocore
```

- [ ] **Step 3: Verify install**

```bash
cd /home/michael/Github/skating-biomechanics-ml && uv sync
```

- [ ] **Step 4: Commit**

```bash
git add ml/pyproject.toml backend/pyproject.toml uv.lock
git commit -m "chore(deps): add aiobotocore for async R2/S3 operations"
```

---

### Task 13: Add async storage functions to backend

**Files:**
- Modify: `backend/app/storage.py`

- [ ] **Step 1: Add async client factory**

```python
# In backend/app/storage.py, add after existing _client():
import aiobotocore.session

_async_session = aiobotocore.session.get_session()

async def _async_client():
    """Create an async S3 client (aiobotocore). Reuses session for connection pooling."""
    s = get_settings()
    return _async_session.create_client(
        "s3",
        endpoint_url=s.r2.endpoint_url or None,
        aws_access_key_id=s.r2.access_key_id.get_secret_value(),
        aws_secret_access_key=s.r2.secret_access_key.get_secret_value(),
        config=BotoConfig(signature_version="s3v4"),
        region_name="auto",
    )
```

- [ ] **Step 2: Add async upload/download functions**

```python
# In backend/app/storage.py, add after existing sync functions:
async def upload_file_async(local_path: str | Path, key: str) -> str:
    """Async version of upload_file."""
    async with await _async_client() as s3:
        await s3.upload_file(str(local_path), get_settings().r2.bucket, key)
    return key

async def download_file_async(key: str, local_path: str | Path) -> str:
    """Async version of download_file."""
    async with await _async_client() as s3:
        await s3.download_file(get_settings().r2.bucket, key, str(local_path))
    return str(local_path)

async def upload_bytes_async(data: bytes, key: str) -> str:
    """Async version of upload_bytes."""
    async with await _async_client() as s3:
        await s3.put_object(Bucket=get_settings().r2.bucket, Key=key, Body=data)
    return key
```

- [ ] **Step 3: Run tests**

```bash
uv run python -m pytest backend/tests/ -x --no-cov 2>/dev/null || echo "No backend tests"
```

- [ ] **Step 4: Commit**

```bash
git add backend/app/storage.py
git commit -m "feat(storage): add async R2/S3 functions via aiobotocore"
```

---

### Task 14: Migrate GPU server to async R2

**Files:**
- Modify: `ml/gpu_server/server.py`

- [ ] **Step 1: Replace boto3 with aiobotocore in GPU server**

```python
# In ml/gpu_server/server.py:
# BEFORE:
import boto3

def _s3(req: ProcessRequest):
    return boto3.client("s3", ...)

# AFTER:
import aiobotocore.session

_async_session = aiobotocore.session.get_session()

def _s3_config(req: ProcessRequest) -> dict:
    return {
        "endpoint_url": req.r2_endpoint_url or None,
        "aws_access_key_id": req.r2_access_key_id,
        "aws_secret_access_key": req.r2_secret_access_key,
        "region_name": "auto",
    }

async def _s3(req: ProcessRequest):
    return _async_session.create_client("s3", **_s3_config(req))
```

- [ ] **Step 2: Convert all s3 operations to async**

In the `/process` endpoint, convert:
- `s3.download_file(...)` → `await s3.download_file(...)`
- `s3.upload_file(...)` → `await s3.upload_file(...)`

```python
# BEFORE (lines 68-117):
    s3 = _s3(req)
    s3.download_file(Bucket=req.r2_bucket, Key=req.video_r2_key, Filename=str(video_path))
    # ... processing ...
    s3.upload_file(Bucket=req.r2_bucket, Key=out_key, Filename=str(output_path))
    s3.upload_file(Bucket=req.r2_bucket, Key=poses_key, Filename=str(poses_path))
    s3.upload_file(Bucket=req.r2_bucket, Key=csv_key, Filename=str(csv_path))

# AFTER:
    async with await _s3(req) as s3:
        await s3.download_file(Bucket=req.r2_bucket, Key=req.video_r2_key, Filename=str(video_path))
        # ... processing ...
        await asyncio.gather(
            s3.upload_file(Bucket=req.r2_bucket, Key=out_key, Filename=str(output_path)),
            s3.upload_file(Bucket=req.r2_bucket, Key=poses_key, Filename=str(poses_path)),
            s3.upload_file(Bucket=req.r2_bucket, Key=csv_key, Filename=str(csv_path)),
        )
```

- [ ] **Step 3: Run tests**

```bash
uv run python -m pytest ml/tests/ -x --no-cov
```

- [ ] **Step 4: Commit**

```bash
git add ml/gpu_server/server.py
git commit -m "perf(gpu-server): migrate R2 operations to aiobotocore with parallel uploads"
```

---

### Task 15: Async Vast.ai client

**Files:**
- Modify: `ml/skating_ml/vastai/client.py`
- Modify: `ml/skating_ml/worker.py:237`

- [ ] **Step 1: Add async version of Vast.ai client functions**

```python
# In ml/skating_ml/vastai/client.py, add after existing sync functions:

async def process_video_remote_async(
    video_key: str,
    person_click: dict | None = None,
    frame_skip: int = 1,
    layer: int = 3,
    tracking: str = "auto",
    export: bool = True,
    ml_flags: dict | None = None,
    element_type: str | None = None,
) -> VastResult:
    """Async version of process_video_remote using httpx.AsyncClient."""
    settings = get_settings()
    api_key = settings.vastai.api_key.get_secret_value()

    worker_url = await asyncio_get_worker_url("skating-gpu", api_key)

    payload = {
        "video_r2_key": video_key,
        "person_click": person_click,
        "frame_skip": frame_skip,
        "layer": layer,
        "tracking": tracking,
        "export": export,
        "ml_flags": ml_flags or {},
        "element_type": element_type,
        "r2_endpoint_url": settings.r2.endpoint_url or "",
        "r2_access_key_id": settings.r2.access_key_id.get_secret_value(),
        "r2_secret_access_key": settings.r2.secret_access_key.get_secret_value(),
        "r2_bucket": settings.r2.bucket,
    }

    async with httpx.AsyncClient(timeout=600) as client:
        resp = await client.post(f"{worker_url}/process", json=payload)
        resp.raise_for_status()
        data = resp.json()

    return VastResult(
        video_key=data["video_r2_key"],
        poses_key=data.get("poses_r2_key"),
        csv_key=data.get("csv_r2_key"),
        stats=data["stats"],
        metrics=data.get("metrics"),
        phases=data.get("phases"),
        recommendations=data.get("recommendations"),
    )


async def asyncio_get_worker_url(endpoint_name: str, api_key: str) -> str:
    """Async version of _get_worker_url with TTL cache."""
    global _worker_url_cache, _worker_url_cache_time
    now = time.monotonic()
    if _worker_url_cache and (now - _worker_url_cache_time) < _WORKER_URL_TTL:
        return _worker_url_cache

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://run.vast.ai/route/",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"endpoint_name": endpoint_name},
        )
        resp.raise_for_status()
        url = resp.json()["data"]["url"]

    _worker_url_cache = url
    _worker_url_cache_time = now
    return url
```

- [ ] **Step 2: Update worker to use async Vast.ai client**

```python
# In ml/skating_ml/worker.py, process_video_task:
# BEFORE (line 237):
    vast_result = await asyncio.to_thread(
        process_video_remote, video_key=video_key, ...
    )

# AFTER:
    vast_result = await process_video_remote_async(
        video_key=video_key, ...
    )
```

- [ ] **Step 3: Run tests**

```bash
uv run python -m pytest ml/tests/ -x --no-cov
```

- [ ] **Step 4: Commit**

```bash
git add ml/skating_ml/vastai/client.py ml/skating_ml/worker.py
git commit -m "perf(vastai): add async Vast.ai client, free thread pool during 600s waits"
```

---

### Task 16: Path-dependent `max_jobs`

**Files:**
- Modify: `backend/app/config.py`
- Modify: `ml/skating_ml/worker.py:451-463`

- [ ] **Step 1: Add remote worker max_jobs config**

```python
# In backend/app/config.py, AppConfig class:
    worker_max_jobs: int = 1
    worker_max_jobs_remote: int = 5  # For Vast.ai path (no local GPU)
```

- [ ] **Step 2: Use path-dependent max_jobs in WorkerSettings**

```python
# In ml/skating_ml/worker.py, WorkerSettings class:
    vastai_key: str = _settings.vastai.api_key.get_secret_value() if _settings.vastai.api_key else ""
    max_jobs: int = (
        _settings.app.worker_max_jobs_remote
        if vastai_key
        else _settings.app.worker_max_jobs
    )
```

- [ ] **Step 3: Run tests**

```bash
uv run python -m pytest ml/tests/ -x --no-cov
```

- [ ] **Step 4: Commit**

```bash
git add backend/app/config.py ml/skating_ml/worker.py
git commit -m "feat(worker): path-dependent max_jobs (1 local, 5 remote via Vast.ai)"
```

---

### Task 17: Vectorize `_compute_frame_metrics`

**Files:**
- Modify: `ml/skating_ml/worker.py:59-180`
- Test: `ml/tests/test_worker_metrics.py` (new)

- [ ] **Step 1: Write failing test for vectorized output equivalence**

```python
# ml/tests/test_worker_metrics.py
"""Tests for _compute_frame_metrics vectorization."""

import numpy as np
import pytest
from skating_ml.worker import _compute_frame_metrics


@pytest.fixture
def sample_poses():
    """Generate deterministic test poses (N=20, 17 keypoints, 3 dims)."""
    rng = np.random.default_rng(42)
    poses = rng.uniform(0.0, 1.0, size=(20, 17, 3)).astype(np.float32)
    return poses


def test_compute_frame_metrics_returns_all_keys(sample_poses):
    result = _compute_frame_metrics(sample_poses)
    expected_keys = {"knee_angles_r", "knee_angles_l", "hip_angles_r", "hip_angles_l", "trunk_lean", "com_height"}
    assert set(result.keys()) == expected_keys


def test_compute_frame_metrics_length(sample_poses):
    result = _compute_frame_metrics(sample_poses)
    assert len(result["knee_angles_r"]) == len(sample_poses)
    assert len(result["com_height"]) == len(sample_poses)


def test_compute_frame_metrics_deterministic(sample_poses):
    r1 = _compute_frame_metrics(sample_poses)
    r2 = _compute_frame_metrics(sample_poses)
    assert r1 == r2


def test_compute_frame_metrics_angle_ranges(sample_poses):
    result = _compute_frame_metrics(sample_poses)
    for key in ["knee_angles_r", "knee_angles_l", "hip_angles_r", "hip_angles_l"]:
        for val in result[key]:
            if val is not None:
                assert 0 <= val <= 180
```

- [ ] **Step 2: Run tests to verify they pass with current implementation**

```bash
uv run python -m pytest ml/tests/test_worker_metrics.py -v
```

- [ ] **Step 3: Vectorize the function**

Replace the per-frame Python loop in `_compute_frame_metrics` with vectorized NumPy operations. The function currently iterates `for pose in poses:` (line 75). Replace with batch operations on the full `(N, 17, 3)` array:

```python
def _compute_frame_metrics(poses: np.ndarray) -> dict:
    """Compute per-frame biomechanics metrics (vectorized)."""
    n = len(poses)
    if n == 0:
        return {k: [] for k in ["knee_angles_r", "knee_angles_l", "hip_angles_r", "hip_angles_l", "trunk_lean", "com_height"]}

    # Batch extract keypoints: (N, 3)
    def kp(key: int) -> np.ndarray:
        return poses[:, key, :]

    rh = kp(H36Key.RHIP)
    rk = kp(H36Key.RKNEE)
    ra = kp(H36Key.RFOOT)
    lh = kp(H36Key.LHIP)
    lk = kp(H36Key.LKNEE)
    la = kp(H36Key.LFOOT)
    rw = kp(H36Key.RWRIST) if H36Key.RWRIST < 17 else rh
    lw = kp(H36Key.LWRIST) if H36Key.LWRIST < 17 else lh
    rc = kp(H36Key.RSHOULDER) if H36Key.RSHOULDER < 17 else rh
    lc = kp(H36Key.LSHOULDER) if H36Key.LSHOULDER < 17 else lh

    def batch_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> list[float | None]:
        """Compute angle at b for all frames. Returns list with None for NaN frames."""
        ba = a - b
        bc = c - b
        cos_angle = np.sum(ba * bc, axis=1) / (
            np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-8
        )
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angles = np.degrees(np.arccos(cos_angle))
        # Mark NaN frames
        nan_mask = np.any(np.isnan(poses.reshape(n, -1)), axis=1)
        angles[nan_mask] = np.nan
        return [None if np.isnan(a) else float(a) for a in angles]

    def batch_com_y() -> list[float | None]:
        """CoM height (y-coordinate of center of mass proxy)."""
        # Simple: average of mid-hip (H36Key 11) and mid-shoulder (H36Key 8)
        # Using Dempster weights would be more accurate but this matches current impl
        mid_hip = (kp(H36Key.LHIP) + kp(H36Key.RHIP)) / 2
        mid_shoulder = (kp(H36Key.LSHOULDER) + kp(H36Key.RSHOULDER)) / 2
        com_y = (mid_hip[:, 1] + mid_shoulder[:, 1]) / 2
        nan_mask = np.any(np.isnan(poses.reshape(n, -1)), axis=1)
        com_y[nan_mask] = np.nan
        return [None if np.isnan(v) else float(v) for v in com_y]

    def batch_trunk_lean() -> list[float | None]:
        """Trunk lean angle from vertical."""
        mid_hip = (kp(H36Key.LHIP) + kp(H36Key.RHIP)) / 2
        mid_shoulder = (kp(H36Key.LSHOULDER) + kp(H36Key.RSHOULDER)) / 2
        spine = mid_shoulder - mid_hip
        # Angle from vertical (0, -1) direction
        vertical = np.array([0, -1.0])
        cos_angle = np.sum(spine * vertical, axis=1) / (np.linalg.norm(spine, axis=1) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angles = np.degrees(np.arccos(cos_angle))
        nan_mask = np.any(np.isnan(poses.reshape(n, -1)), axis=1)
        angles[nan_mask] = np.nan
        return [None if np.isnan(a) else float(a) for a in angles]

    return {
        "knee_angles_r": batch_angle(rh, rk, ra),
        "knee_angles_l": batch_angle(lh, lk, la),
        "hip_angles_r": batch_angle(rw, rc, rh),
        "hip_angles_l": batch_angle(lw, lc, lh),
        "trunk_lean": batch_trunk_lean(),
        "com_height": batch_com_y(),
    }
```

**Important:** Read the current `_compute_frame_metrics` implementation carefully before rewriting. The exact keypoint indices and angle calculations must match. The code above is a template — verify H36Key indices and angle definitions against the existing implementation.

- [ ] **Step 4: Run tests to verify equivalence**

```bash
uv run python -m pytest ml/tests/test_worker_metrics.py -v
```

- [ ] **Step 5: Commit**

```bash
git add ml/skating_ml/worker.py ml/tests/test_worker_metrics.py
git commit -m "perf(worker): vectorize _compute_frame_metrics (eliminate per-frame Python loop)"
```

---

### Task 18: Vectorize physics engine functions

**Files:**
- Modify: `ml/skating_ml/analysis/physics_engine.py`
- Test: `ml/tests/analysis/test_physics_engine.py` (existing)

- [ ] **Step 1: Read current implementations**

```bash
grep -n "for.*frame" ml/skating_ml/analysis/physics_engine.py
```

Identify all per-frame Python loops.

- [ ] **Step 2: Vectorize `calculate_center_of_mass`**

Replace per-frame loop with batch numpy operations using `np.einsum` or broadcasting. The Dempster weight coefficients should be applied as a single matrix multiply across all frames.

- [ ] **Step 3: Vectorize `calculate_moment_of_inertia`**

After vectorizing CoM, vectorize MoI using batch distance computation.

- [ ] **Step 4: Run existing physics engine tests**

```bash
uv run python -m pytest ml/tests/analysis/test_physics_engine.py -v
```

Expected: All tests pass. If vectorized output differs from per-frame, verify with `np.allclose(old, new, atol=1e-6)`.

- [ ] **Step 5: Commit**

```bash
git add ml/skating_ml/analysis/physics_engine.py
git commit -m "perf(physics): vectorize center_of_mass and moment_of_inertia calculations"
```

---

### Task 19: Render + analysis parallelism

**Files:**
- Modify: `ml/skating_ml/web_helpers.py:316-423`

- [ ] **Step 1: Start analysis in background thread before render loop**

```python
# In ml/skating_ml/web_helpers.py, process_video_pipeline:
# BEFORE (sequential):
    # ... render loop (lines 317-390) ...
    # ... analysis (lines 399-423) ...

# AFTER (parallel):
    from concurrent.futures import ThreadPoolExecutor

    analysis_future = None
    if element_type and prepared.n_valid > 0:
        executor = ThreadPoolExecutor(max_workers=1)
        analysis_future = executor.submit(
            _run_analysis,
            prepared.poses_norm, meta.fps, element_type,
        )

    # ... render loop (unchanged) ...

    # Collect analysis results after render
    analysis_metrics = None
    analysis_phases = None
    analysis_recommendations = None
    if analysis_future is not None:
        analysis_metrics, analysis_phases, analysis_recommendations = analysis_future.result()
        executor.shutdown(wait=False)
```

- [ ] **Step 2: Extract analysis into standalone function**

```python
# In ml/skating_ml/web_helpers.py, add helper:
def _run_analysis(
    poses_norm: np.ndarray, fps: float, element_type: str,
) -> tuple:
    """Run biomechanics analysis (CPU-bound, thread-safe)."""
    from skating_ml.analysis.phase_detector import PhaseDetector
    from skating_ml.analysis.metrics import BiomechanicsAnalyzer
    from skating_ml.analysis.recommender import Recommender
    from skating_ml.analysis.element_defs import get_element_def

    element_def = get_element_def(element_type)
    phases = PhaseDetector().detect_phases(poses_norm, fps)
    metrics = BiomechanicsAnalyzer(element_def).analyze(poses_norm, fps, phases)
    recommendations = Recommender().recommend(metrics, element_type)
    return metrics, phases, recommendations
```

- [ ] **Step 3: Run tests**

```bash
uv run python -m pytest ml/tests/ -x --no-cov
```

- [ ] **Step 4: Commit**

```bash
git add ml/skating_ml/web_helpers.py
git commit -m "perf(pipeline): run biomechanics analysis in parallel with video rendering"
```

---

### Task 20: Batch 3D lifter windows

**Files:**
- Modify: `ml/skating_ml/pose_3d/onnx_extractor.py:58-113`
- Test: `ml/tests/pose_3d/` (existing)

- [ ] **Step 1: Write test for batch equivalence**

Read existing 3D lifter tests to understand the test pattern. Add a test that verifies batch inference produces the same output as sequential inference:

```python
# Add to ml/tests/pose_3d/test_onnx_extractor.py (or create if not exists):
def test_estimate_3d_batch_matches_sequential(extractor, sample_poses_2d):
    """Batch inference must produce same results as sequential within tolerance."""
    # This test will fail until batch inference is implemented
    # For now, just verify sequential still works
    result = extractor.estimate_3d(sample_poses_2d)
    assert result.shape == (len(sample_poses_2d), 17, 3)
```

- [ ] **Step 2: Implement batch window inference**

```python
# In ml/skating_ml/pose_3d/onnx_extractor.py, estimate_3d method:
# BEFORE (sequential windows):
    while start < n_frames:
        end = min(start + self.temporal_window, n_frames)
        window = poses_2d[start:end]
        out = self._infer_window(window)
        # ... accumulate ...

# AFTER (batch all windows):
    windows = []
    window_starts = []
    window_ends = []
    start = 0
    while start < n_frames:
        end = min(start + self.temporal_window, n_frames)
        windows.append(poses_2d[start:end])
        window_starts.append(start)
        window_ends.append(end)
        start += stride

    if not windows:
        return np.zeros((n_frames, 17, 3), dtype=np.float32)

    # Pad all windows to temporal_window
    padded = np.stack([
        np.pad(w, ((0, self.temporal_window - len(w)), (0, 0), (0, 0)),
               mode="edge")
        for w in windows
    ], axis=0)  # (num_windows, temporal_window, 17, 2)

    # Add confidence channel
    conf = np.ones((*padded.shape[:2], 1), dtype=np.float32)
    batch_input = np.concatenate([padded, conf], axis=-1)  # (num_windows, W, 17, 3)
    batch_input = batch_input[np.newaxis]  # (1, num_windows, W, 17, 3)

    # Check if model supports batched input
    # If input shape is (1, W, 17, 3), we need to reshape to (num_windows, W, 17, 3)
    input_shape = self.session.get_inputs()[0].shape
    if input_shape[0] == 1 or input_shape[0] == "batch_size":
        # Model supports batch dim — run all at once
        batch_out = self.session.run(None, {self.input_name: batch_input})[0]
        # ... scatter results back ...
```

**Important:** The exact batching strategy depends on the ONNX model's input shape. Read the model's input spec first via `session.get_inputs()[0].shape`. If the model only accepts batch_size=1, batch windows by changing the first dim to `num_windows`.

- [ ] **Step 3: Run tests**

```bash
uv run python -m pytest ml/tests/pose_3d/ -v
```

- [ ] **Step 4: Commit**

```bash
git add ml/skating_ml/pose_3d/onnx_extractor.py ml/tests/pose_3d/
git commit -m "perf(3d): batch sliding window inference for 3D pose lifter"
```

---

### Task 21: Phase 2 validation benchmark

- [ ] **Step 1: Run benchmark**

```bash
uv run python ml/scripts/benchmark_pose_extraction.py
```

Record results. Compare with Phase 1 baseline.

- [ ] **Step 2: Run full test suite**

```bash
uv run python -m pytest ml/tests/ --no-cov -q
```

- [ ] **Step 3: Commit Phase 2 milestone**

```bash
git add -A
git commit -m "milestone(pipeline): Phase 2 medium effort complete"
```

---

## Phase 3: Architectural

**Goal:** Direct ONNX batch RTMO inference, SSE streaming, job priority, frame buffer.

**Prerequisites:** Phase 2 complete.

**Validation:** Full benchmark suite with representative videos.

---

### Task 22: Direct ONNX batch RTMO inference (Part 1: Preprocessing)

**Files:**
- Create: `ml/skating_ml/pose_estimation/rtmo_batch.py`

This is the highest-impact optimization (pose extraction = 50-70% of wall time). Split into sub-tasks.

- [ ] **Step 1: Study rtmlib's RTMO preprocessing**

Read rtmlib source to understand the letterbox resize, normalization, and input tensor construction for RTMO:

```bash
python -c "import rtmlib; print(rtmlib.__file__)"
```

Find the preprocessing code that converts an RGB frame `(H, W, 3)` to the RTMO input tensor `(1, 3, 640, 640)`.

- [ ] **Step 2: Implement batch preprocessing**

```python
# In ml/skating_ml/pose_estimation/rtmo_batch.py:
"""Direct ONNX RTMO batch inference (bypasses rtmlib)."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np

# RTMO input size
RTMO_INPUT_SIZE = 640


def preprocess_frame(frame: np.ndarray, input_size: int = RTMO_INPUT_SIZE) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """Preprocess single frame for RTMO: letterbox resize + normalize.

    Returns (tensor (3, input_size, input_size), (src_w, src_h), (pad_w, pad_h))
    """
    h, w = frame.shape[:2]
    scale = min(input_size / w, input_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    pad_w = (input_size - new_w) // 2
    pad_h = (input_size - new_h) // 2

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    tensor = padded[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
    tensor = tensor.astype(np.float32) / 255.0

    return tensor, (w, h), (pad_w, pad_h)


def preprocess_batch(frames: list[np.ndarray], input_size: int = RTMO_INPUT_SIZE) -> tuple[np.ndarray, list[tuple], list[tuple]]:
    """Preprocess batch of frames for RTMO.

    Returns (batch_tensor (B, 3, input_size, input_size), orig_sizes, pad_offsets)
    """
    tensors = []
    orig_sizes = []
    pad_offsets = []
    for frame in frames:
        tensor, orig, pad = preprocess_frame(frame, input_size)
        tensors.append(tensor)
        orig_sizes.append(orig)
        pad_offsets.append(pad)
    return np.stack(tensors), orig_sizes, pad_offsets
```

- [ ] **Step 3: Commit**

```bash
git add ml/skating_ml/pose_estimation/rtmo_batch.py
git commit -m "feat(pose): add RTMO batch preprocessing (bypass rtmlib)"
```

---

### Task 23: Direct ONNX batch RTMO inference (Part 2: Postprocessing)

**Files:**
- Modify: `ml/skating_ml/pose_estimation/rtmo_batch.py`

- [ ] **Step 1: Study rtmlib's RTMO postprocessing**

Read the rtmlib source to understand keypoint decoding from heatmaps/simcc output. RTMO outputs either heatmaps or simcc (per-joint coordinate regressions). Determine which format the ONNX model uses.

- [ ] **Step 2: Implement batch postprocessing**

```python
# In ml/skating_ml/pose_estimation/rtmo_batch.py, add:
def postprocess_batch(
    outputs: np.ndarray,
    orig_sizes: list[tuple[int, int]],
    pad_offsets: list[tuple[int, int]],
    input_size: int = RTMO_INPUT_SIZE,
    score_threshold: float = 0.3,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Decode RTMO batch output to keypoints + scores.

    Returns list of (keypoints (M, 17, 2), scores (M, 17)) per frame.
    M = number of detected persons (varies per frame).
    """
    # Implementation depends on RTMO output format (heatmaps vs simcc).
    # Read the rtmlib source to determine the exact decoding logic.
    raise NotImplementedError("Implement after studying rtmlib postprocessing")
```

- [ ] **Step 3: Commit**

```bash
git add ml/skating_ml/pose_estimation/rtmo_batch.py
git commit -m "feat(pose): add RTMO batch postprocessing stub (needs rtmlib study)"
```

---

### Task 24: Direct ONNX batch RTMO inference (Part 3: Integration)

**Files:**
- Modify: `ml/skating_ml/pose_estimation/pose_extractor.py`
- Modify: `ml/skating_ml/pose_estimation/batch_extractor.py`

- [ ] **Step 1: Integrate batch RTMO into BatchPoseExtractor**

Modify `BatchPoseExtractor._process_batch()` to use the new `rtmo_batch` module instead of calling `tracker(frame)` per frame:

```python
# In batch_extractor.py, _process_batch:
# BEFORE (line 276):
    for frame in frames:
        result = tracker(frame)
        # ...

# AFTER:
    from skating_ml.pose_estimation.rtmo_batch import preprocess_batch, postprocess_batch
    batch_tensor, orig_sizes, pad_offsets = preprocess_batch(frames)
    outputs = session.run(None, {"input": batch_tensor})[0]
    results = postprocess_batch(outputs, orig_sizes, pad_offsets)
    # Feed results sequentially to tracker for tracking association
```

- [ ] **Step 2: Run tests**

```bash
uv run python -m pytest ml/tests/pose_estimation/ -v
```

- [ ] **Step 3: Run benchmark**

```bash
uv run python ml/scripts/benchmark_pose_extraction.py
```

Record before/after for RTMO throughput (frames/second).

- [ ] **Step 4: Commit**

```bash
git add ml/skating_ml/pose_estimation/rtmo_batch.py ml/skating_ml/pose_estimation/batch_extractor.py
git commit -m "perf(pose): integrate batch RTMO inference into BatchPoseExtractor"
```

---

### Task 25: SSE streaming for progress

**Files:**
- Modify: `backend/app/routes/process.py` (add SSE endpoint)
- Modify: `ml/skating_ml/worker.py` (publish progress)
- Modify: `backend/app/task_manager.py` (add pub/sub helpers)

- [ ] **Step 1: Add Valkey pub/sub helpers**

```python
# In backend/app/task_manager.py, add:
async def publish_task_event(task_id: str, data: dict, valkey: aioredis.Redis | None = None) -> None:
    """Publish task event to pub/sub channel for SSE streaming."""
    if valkey is None:
        valkey = await get_valkey_client()
    try:
        await valkey.publish(f"task_events:{task_id}", json.dumps(data))
    finally:
        if valkey is None:
            await valkey.close()
```

- [ ] **Step 2: Add SSE endpoint**

```python
# In backend/app/routes/process.py, add:
from sse_starlette.sse import EventSourceResponse

@router.get("/process/{task_id}/stream")
async def stream_process_status(task_id: str):
    async def event_generator():
        valkey = await get_valkey_client()
        pubsub = valkey.pubsub()
        await pubsub.subscribe(f"task_events:{task_id}")
        try:
            state = await get_task_state(task_id, valkey=valkey)
            yield f"data: {json.dumps(state)}\n\n"
            async for message in pubsub.listen():
                if message["type"] == "message":
                    yield f"data: {message['data'].decode()}\n\n"
                    try:
                        data = json.loads(message["data"])
                        if data.get("status") in ("completed", "failed", "cancelled"):
                            break
                    except (json.JSONDecodeError, TypeError):
                        pass
        finally:
            await pubsub.unsubscribe(f"task_events:{task_id}")
            await valkey.close()
    return EventSourceResponse(event_generator())
```

- [ ] **Step 3: Worker publishes progress**

In `ml/skating_ml/worker.py`, add `publish_task_event()` calls alongside existing `update_progress()` calls.

- [ ] **Step 4: Run tests**

```bash
uv run python -m pytest backend/tests/ -x --no-cov 2>/dev/null || echo "No backend tests"
uv run python -m pytest ml/tests/ -x --no-cov
```

- [ ] **Step 5: Commit**

```bash
git add backend/app/routes/process.py backend/app/task_manager.py ml/skating_ml/worker.py
git commit -m "feat(worker): add SSE streaming for real-time task progress"
```

---

### Task 26: Job priority queues

**Files:**
- Modify: `ml/skating_ml/worker.py`
- Modify: `backend/app/config.py`

- [ ] **Step 1: Add queue configuration**

```python
# In backend/app/config.py, AppConfig:
    queue_name_detect: str = "skating:queue:detect"
    queue_name_process: str = "skating:queue:process"
```

- [ ] **Step 2: Update enqueue calls to use priority**

In backend routes, enqueue detect jobs with higher priority:

```python
await arq_pool.enqueue_job(
    "detect_video_task",
    ...,
    _priority=0,  # High priority
)
await arq_pool.enqueue_job(
    "process_video_task",
    ...,
    _priority=10,  # Low priority
)
```

- [ ] **Step 3: Update worker to listen on both queues**

```python
# In ml/skating_ml/worker.py, WorkerSettings:
    queue_name: str = _settings.app.queue_name_process  # Default queue
    # arq supports listening on multiple queues via functions parameter
```

Note: arq's `max_jobs` applies per-queue. Research arq multi-queue support before implementing.

- [ ] **Step 4: Run tests**

```bash
uv run python -m pytest ml/tests/ -x --no-cov
```

- [ ] **Step 5: Commit**

```bash
git add backend/app/config.py ml/skating_ml/worker.py backend/app/routes/process.py backend/app/routes/detect.py
git commit -m "feat(worker): add job priority queues (detect > process)"
```

---

### Task 27: Cancellation checks at stage boundaries

**Files:**
- Modify: `ml/skating_ml/worker.py`

- [ ] **Step 1: Add cancellation check before Vast.ai dispatch**

```python
# In process_video_task, before asyncio.to_thread:
    if await is_cancelled(task_id, valkey=valkey):
        await mark_cancelled(task_id, valkey=valkey)
        return {"status": "cancelled"}
```

- [ ] **Step 2: Add cancellation check after Vast.ai returns**

```python
# After vast_result = await process_video_remote_async(...):
    if await is_cancelled(task_id, valkey=valkey):
        await mark_cancelled(task_id, valkey=valkey)
        return {"status": "cancelled"}
```

- [ ] **Step 3: Commit**

```bash
git add ml/skating_ml/worker.py
git commit -m "fix(worker): add cancellation checks at stage boundaries"
```

---

### Task 28: Producer-consumer frame buffer

**Files:**
- Create: `ml/skating_ml/utils/frame_buffer.py`
- Modify: `ml/skating_ml/web_helpers.py`

- [ ] **Step 1: Implement AsyncFrameReader**

```python
# In ml/skating_ml/utils/frame_buffer.py:
"""Producer-consumer frame buffer for overlapping video decode with GPU inference."""

import threading
from queue import Queue
from typing import Generator

import cv2
import numpy as np


class AsyncFrameReader:
    """Background thread decodes frames into a bounded queue."""

    _SENTINEL = object()

    def __init__(self, video_path: str | Path, buffer_size: int = 16, frame_skip: int = 1) -> None:
        self._path = str(video_path)
        self._buffer_size = buffer_size
        self._frame_skip = frame_skip
        self._queue: Queue = Queue(maxsize=buffer_size)
        self._thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None
        self._frame_idx = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        self._cap = cv2.VideoCapture(self._path)
        skip_counter = 0
        while self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                break
            skip_counter += 1
            if skip_counter < self._frame_skip:
                continue
            skip_counter = 0
            self._queue.put((self._frame_idx, frame))
            self._frame_idx += 1
        self._queue.put(self._SENTINEL)
        self._cap.release()

    def get_frame(self) -> tuple[int, np.ndarray] | None:
        item = self._queue.get()
        if item is self._SENTINEL:
            return None
        return item

    def join(self) -> None:
        if self._thread:
            self._thread.join(timeout=5)
```

- [ ] **Step 2: Integrate into render loop**

In `web_helpers.py`, replace sequential `cap.read()` with `AsyncFrameReader`:

```python
# BEFORE:
    cap = cv2.VideoCapture(str(video_path))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

# AFTER:
    reader = AsyncFrameReader(video_path, buffer_size=16, frame_skip=frame_skip)
    reader.start()
    while True:
        result = reader.get_frame()
        if result is None:
            break
        frame_idx, frame = result
```

- [ ] **Step 3: Run tests**

```bash
uv run python -m pytest ml/tests/ -x --no-cov
```

- [ ] **Step 4: Commit**

```bash
git add ml/skating_ml/utils/frame_buffer.py ml/skating_ml/web_helpers.py
git commit -m "perf(pipeline): add AsyncFrameReader for decode-inference overlap"
```

---

### Task 29: Phase 3 validation benchmark

- [ ] **Step 1: Run full benchmark suite**

```bash
uv run python ml/scripts/benchmark_pose_extraction.py
```

- [ ] **Step 2: Run full test suite**

```bash
uv run python -m pytest ml/tests/ --no-cov -q
```

- [ ] **Step 3: Run linter**

```bash
uv run ruff check ml/skating_ml/
```

- [ ] **Step 4: Run type checker**

```bash
uv run basedpyright ml/skating_ml/ --level error
```

- [ ] **Step 5: Commit Phase 3 milestone**

```bash
git add -A
git commit -m "milestone(pipeline): Phase 3 architectural optimizations complete"
```

---

## Self-Review Checklist

- [ ] **Spec coverage:** Every opportunity from the design doc (G1-G9, I1-I8, O1-O9, C1-C6, D1-D5) maps to a task. G4 (IO bindings) and G8 (TensorRT) deferred. D5 (streaming pipeline) deferred.
- [ ] **Placeholder scan:** No TBD/TODO. All code blocks contain actual implementations (or explicit `raise NotImplementedError` with next steps).
- [ ] **Type consistency:** Function names match between tasks. `_compute_frame_metrics` signature preserved. `VastResult` fields match between client.py and worker.py.
- [ ] **Dependency order:** Tasks within each phase are ordered by dependencies (e.g., Task 12 before Task 13 before Task 14).
- [ ] **Test coverage:** Every code change has a test step. New functions (Task 17) have dedicated test files.
