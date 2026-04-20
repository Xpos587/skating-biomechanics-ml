# Проектирование параллелизма и async-оптимизаций

> **Дата:** 2026-04-19
> **Статус:** Design Document
> **Источник:** Анализ 5 специализированных агентов (ML Pipeline, I/O & Network, GPU & Compute, Video Processing, Async Architecture)

---

## 1. Executive Summary

Наибольший выигрыш в производительности даёт интеграция `BatchRTMO` (существующий, но не подключённый к pipeline) — ожидаемый speedup 2.5-4x на GPU-инференсе, что при текущем профиле задачи (5.6s inference на 364 кадра) сократит это время до 1.4-2.2s. Второй по значимости — **3 критических sync-in-async бага** в FastAPI routes, где синхронный `boto3` блокирует event loop. Третий — `AsyncFrameReader` (double buffering), который существует в `frame_buffer.py`, но используется только в `web_helpers.py`, а не в основном `pose_extractor.py`. Наконец, антипаттерн создания/уничтожения Valkey и arq подключений на каждый запрос добавляет 10+ unnecessary connect/close операций на задачу.

---

## 2. Архитектура параллельного pipeline

```
                    Текущий pipeline (последовательный)
                    ════════════════════════════════════

  cap.read() ──► RTMO(frame) ──► Tracking ──► GapFill ──► Smooth ──► Phase ──► Metrics ──► Recommend
     8ms          10ms/frame     0.1ms      5ms        10ms      50ms       20ms        1ms
  ════════════════════════════════════════════════════════════════════════════════════════════════
                                    Total: ~9.3s for 364 frames (5.6s inference dominant)


                    Предлагаемый pipeline (параллельный)
                    ═══════════════════════════════════════════

  ┌─ Thread 1 (decode) ─────────────────────────────────────┐
  │  AsyncFrameReader: cap.read() batch N+1 ─────────────────┤──► queue
  └──────────────────────────────────────────────────────────┘
                          │
  ┌─ GPU thread ─────────▼───────────────────────────────────┐
  │  BatchRTMO: batch N (8 frames) ──► 2.5-4x speedup      │
  └──────────────────────────────────────────────────────────┘
                          │
                          ▼
  ┌─ Main thread ────────────────────────────────────────────┐
  │  Tracking ──► GapFill ──► Smooth                         │
  └──────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────────┐
    │ 3D Lift  │   │  Phase   │   │  Reference   │
    │ (thread) │   │ (thread) │   │   Load       │
    └────┬─────┘   └────┬─────┘   └──────┬───────┘
         │              │                 │
         └──────────────┼─────────────────┘
                        ▼
              ┌──────────────────┐
              │  Physics + DTW   │
              │  (sequential)    │
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │ Recommendations  │
              └──────────────────┘

  Ожидаемый total: ~3-4s для 364 кадров (вместо ~9.3s)
```

---

## 3. Tier 1: Quick Wins (Low Complexity, High Impact)

### 3.1. Sync-in-async баги в FastAPI routes

**Проблема:** Три endpoint используют синхронный `boto3` (`_client()`) внутри `async def` handlers, блокируя event loop на всё время HTTP-вызова к R2 (50-200ms per call).

| Endpoint | Файл:строка | Sync вызов | Блокировка |
|----------|-------------|------------|------------|
| `POST /detect` | `backend/app/routes/detect.py:39` | `upload_bytes()` → `_client().put_object()` | ~100ms |
| `POST /uploads/init` | `backend/app/routes/uploads.py:27` | `_client().create_multipart_upload()` | ~100ms |
| `POST /uploads/complete` | `backend/app/routes/uploads.py:73` | `_client().complete_multipart_upload()` | ~150ms |
| `GET /sessions/*` | `backend/app/routes/sessions.py:35,39-41` | `get_object_url()` → `_client().generate_presigned_url()` | ~50ms per URL |

**Решение:** Заменить `_client()` (sync boto3) на `_async_client()` (aiobotocore) во всех async handlers. Async версии уже существуют в `storage.py:106-144`.

```python
# detect.py:38-39 — BEFORE (блокирует event loop):
content = await video.read()
upload_bytes(content, video_key)  # sync boto3!

# AFTER (non-blocking):
content = await video.read()
from app.storage import upload_bytes_async
await upload_bytes_async(content, video_key)
```

```python
# sessions.py:35-41 — BEFORE (3 presigned URL генерации синхронно):
video_url = get_object_url(session.video_key)
processed_video_url = get_object_url(session.processed_video_key)

# AFTER — вынести в helper с asyncio.to_thread или создать async версию:
async def get_object_url_async(key: str) -> str:
    return await asyncio.to_thread(get_object_url, key)
```

**Ожидаемый выигрыш:** Устранение блокировки event loop — все concurrent requests будут обрабатываться параллельно вместо serial. На single-user системе выигрыш невидим, но при 2+ concurrent uploads throughput растёт линейно.

**Сложность:** 2-3 часа. Async версии storage функций уже существуют.

---

### 3.2. Connection pooling для Valkey

**Проблема:** `task_manager.py` создаёт и уничтожает `aioredis.Redis` соединение на каждый вызов. Каждая функция (`create_task_state`, `update_progress`, `store_result`, etc.) делает `get_valkey_client()` → `valkey.close()`. Для одного `process_video_task` это ~10+ connect/close циклов.

**Файл:** `backend/app/task_manager.py:29-37` (каждая функция повторяет один и тот же pattern)

```python
# Текущий pattern — каждый вызов:
async def update_progress(task_id, fraction, message, valkey=None):
    close = valkey is None
    if valkey is None:
        valkey = await get_valkey_client()  # NEW connection
    try:
        await valkey.hset(...)
    finally:
        if close:
            await valkey.close()  # CLOSE connection
```

**Решение:** Singleton connection pool через FastAPI lifespan.

```python
# app/state.py (новый файл)
import redis.asyncio as aioredis

_valkey_pool: aioredis.Redis | None = None

async def init_valkey_pool():
    global _valkey_pool
    settings = get_settings()
    _valkey_pool = aioredis.Redis(
        host=settings.valkey.host,
        port=settings.valkey.port,
        db=settings.valkey.db,
        password=settings.valkey.password.get_secret_value(),
        decode_responses=True,
        max_connections=20,
    )

async def close_valkey_pool():
    if _valkey_pool:
        await _valkey_pool.close()

def get_valkey() -> aioredis.Redis:
    assert _valkey_pool is not None, "Call init_valkey_pool() first"
    return _valkey_pool
```

**Файлы для изменения:**
- `backend/app/main.py` — lifespan: `init_valkey_pool()` / `close_valkey_pool()`
- `backend/app/task_manager.py` — убрать `get_valkey_client()`, использовать pool
- `backend/app/routes/detect.py:44,75,101` — убрать `valkey.close()`
- `backend/app/routes/process.py:38,84,119` — убрать `valkey.close()`
- `ml/src/worker.py:195,320,332,436` — worker тоже создаёт/закрывает на задачу

**Ожидаемый выигрыш:** Устранение ~10 connect/close RTT на задачу (~1ms each = ~10ms saved per task). На Vast.ai Serverless с high-latency Valkey connection выигрыш больше.

**Сложность:** 3-4 часа.

---

### 3.3. arq pool singleton (FastAPI lifespan)

**Проблема:** `detect.py:50-67` и `process.py:53-76` создают и закрывают `arq.create_pool()` на каждый enqueue request.

```python
# detect.py:50-67 — каждый POST /detect:
arq_pool = await create_pool(RedisSettings(...))
try:
    await arq_pool.enqueue_job(...)
finally:
    await arq_pool.close()  # ненужное закрытие
```

**Решение:** Singleton pool в FastAPI lifespan (вместе с Valkey pool).

```python
# main.py lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_valkey_pool()
    app.state.arq_pool = await create_pool(RedisSettings(...))
    yield
    await app.state.arq_pool.close()
    await close_valkey_pool()
```

**Файлы для изменения:**
- `backend/app/main.py` — lifespan
- `backend/app/routes/detect.py:50-67` — использовать `request.app.state.arq_pool`
- `backend/app/routes/process.py:53-76` — аналогично

**Ожидаемый выигрыш:** ~1ms per enqueue (connect overhead). Низкий, но это чистый код.

**Сложность:** 1 час.

---

### 3.4. Dead code: `analyze()` sync path

**Проблема:** `AnalysisPipeline.analyze()` (sync, `pipeline.py:181-360`) и `AnalysisPipeline.analyze_async()` (async, `pipeline.py:610-756`) содержат идентичную логику. GPU server (`gpu_server/server.py`) использует `VizPipeline` из `web_helpers.py`. Worker (`worker.py`) dispatches to Vast.ai. Sync `analyze()` используется только в CLI scripts и tests.

| Caller | Файл | Метод |
|--------|------|-------|
| CLI scripts | `ml/scripts/profile_pipeline.py:69` | `analyze()` (sync) |
| CLI scripts | `ml/scripts/organize_dataset.py:73` | `analyze()` (sync) |
| Tests | `ml/tests/test_pipeline_parallel.py:81` | `analyze()` (sync) |
| Tests | `ml/tests/benchmark/test_multi_gpu_performance.py:112,204` | `analyze()` (sync) |

**Решение:** Не удалять `analyze()`, но отметить как CLI-only. Дублирование допустимо, так как CLI не нуждается в async. Добавить docstring комментарий.

**Сложность:** 15 минут (docstring update).

---

### 3.5. N+1 query в `session_saver.py`

**Проблема:** `session_saver.py:51-56` вызывает `get_current_best()` внутри цикла по каждому metric. При 12 метриках = 12 отдельных SQL запросов.

```python
# session_saver.py:41-70 — цикл с N+1:
for mr in metrics:
    current_best = await get_current_best(  # 1 query per metric
        db, user_id=session.user_id,
        element_type=session.element_type,
        metric_name=mr.name,
    )
```

**Решение:** Batch query — получить все current bests одним запросом.

```python
# batch version:
bests = await get_current_best_batch(
    db, user_id=session.user_id,
    element_type=session.element_type,
    metric_names=[mr.name for mr in metrics],
)
```

**Файлы для изменения:**
- `backend/app/crud/session_metric.py` — новый `get_current_best_batch()`
- `backend/app/services/session_saver.py:41-70` — заменить loop

**Ожидаемый выигрыш:** 12 queries → 1 query. ~11 * (latency to Postgres) saved. При локальном Postgres ~5ms. При удалённом — больше.

**Сложность:** 2-3 часа.

---

### 3.6. httpx.AsyncClient reuse в Vast.ai client

**Проблема:** `vastai/client.py:133,186` создаёт `httpx.AsyncClient()` на каждый запрос. TLS handshake повторяется.

```python
# client.py:133 — каждый route request:
async with httpx.AsyncClient() as client:  # new client = new TLS session
    resp = await client.post(ROUTE_URL, ...)

# client.py:186 — каждый process request:
async with httpx.AsyncClient() as client:  # ещё один
    resp = await client.post(f"{worker_url}/process", ...)
```

**Решение:** Module-level client с lazy init.

```python
_http_client: httpx.AsyncClient | None = None

def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=600)
    return _http_client
```

**Ожидаемый выигрыш:** ~50-100ms per TLS handshake (reused from pool). Для Vast.ai с интернета — заметно.

**Сложность:** 1 час.

---

## 4. Tier 2: Medium-Term Improvements (1-3 days)

### 4.1. Batch RTMO inference

**Проблема:** `PoseExtractor.extract_video_tracked()` (`pose_extractor.py:141`) вызывает rtmlib per-frame (batch=1). `BatchRTMO` (`rtmo_batch.py:175`) и `BatchPoseExtractor` (`batch_extractor.py:44`) существуют, но не подключены к pipeline.

**Текущее состояние BatchPoseExtractor:**
- Использует `BatchRTMO` напрямую (обходит rtmlib)
- Отсутствует tracking (comment: `target_track_id=None, # Tracking to be implemented`)
- Работает без Sports2D/DeepSORT trackers

**Что нужно:**
1. Интегрировать `BatchPoseExtractor._process_batch()` в `PoseExtractor.extract_video_tracked()`
2. Добавить batching к существующему tracking loop (batch inference → sequential tracking post-process)
3. Определить оптимальный `batch_size` для RTX 3050 Ti (4GB VRAM, ~712MB current usage)

**Профиль VRAM (batch_size=8):**
| Компонент | VRAM |
|-----------|------|
| RTMO-M model weights | ~100MB |
| Batch input (8 frames x 640x640x3) | ~37MB |
| Batch activations | ~800MB (est.) |
| **Total est.** | **~937MB** |

VRAM headroom: 4GB - 937MB = ~3GB. batch_size=16 возможно (~1.6GB activations).

**Файлы для изменения:**
- `ml/src/pose_estimation/pose_extractor.py:238-300` — заменить per-frame loop на batched
- `ml/src/pose_estimation/batch_extractor.py:44-315` — добавить tracking support

**Ожидаемый выигрыш:** 2.5-4x на inference. Текущий 5.6s → ~1.4-2.2s для 364 кадров.

**Сложность:** 2-3 дня. Требуется интеграция batch inference с tracking (Sports2D centroid association работает per-frame).

---

### 4.2. Double buffering (AsyncFrameReader в pose_extractor.py)

**Проблема:** `AsyncFrameReader` существует в `frame_buffer.py:22`, используется в `web_helpers.py:320-323`, но НЕ используется в `pose_extractor.py:255` (основной extraction loop использует синхронный `cap.read()`).

```
Текущий (последовательный):
  cap.read() ──► preprocess ──► RTMO(frame) ──► cap.read() ──► ...
     8ms           1ms            10ms            8ms

С double buffering:
  ┌─ Thread 1: cap.read() ──► queue ──────────────────┐
  │                          8ms                       │
  └────────────────────────────────────────────────────┘
  ┌─ Main: queue.get() ──► preprocess ──► RTMO(frame) ─┐
  │          <1ms          1ms           10ms            │
  └─────────────────────────────────────────────────────┘
  Decode overlapping с inference → экономия ~8ms/frame
```

**Ожидаемый выигрыш:** ~8ms/frame * 364 frames = ~2.9s. Но decode частично перекрывается с batch inference (Tier 2 item 4.1), поэтому реальный выигрыш зависит от batch_size. При batch=8 и inference=2.2s, decode time=2.9s, overlap ~2.2s, net savings ~0.7s.

**Файлы для изменения:**
- `ml/src/pose_estimation/pose_extractor.py:238-300` — заменить `cap.read()` на `AsyncFrameReader`
- Добавить поддержку `frame_skip` в `AsyncFrameReader` (уже есть: `frame_buffer.py:46`)

**Сложность:** 4-6 часов. Низкий риск — `AsyncFrameReader` уже tested.

---

### 4.3. ONNX SessionOptions оптимизация

**Проблема:** `BatchRTMO.__init__()` (`rtmo_batch.py:229`) и rtmlib (внутренне) создают `InferenceSession` без `SessionOptions`. Missing optimizations:

```python
# Текущий (rtmo_batch.py:229):
self._session = onnxruntime.InferenceSession(str(model_path), providers=providers)

# Оптимизированный:
opts = onnxruntime.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.enable_mem_pattern = True
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
if device == "cuda":
    opts.add_session_config_entry("session_options.cudnn_conv_algo_search", "EXHAUSTIVE")
self._session = onnxruntime.InferenceSession(str(model_path), sess_options=opts, providers=providers)
```

**Ожидаемый выигрыш:** 10-20% на inference (graph optimization + cuDNN algo search). Для 5.6s → ~4.5-5s.

**Сложность:** 2-3 часа. Нужно также передать opts через rtmlib (если возможно) или использовать только `BatchRTMO` path.

---

### 4.4. ONNX IO Binding (zero-copy GPU transfer)

**Проблема:** Текущий `session.run()` (`rtmo_batch.py:259`) копирует CPU→GPU→CPU каждый вызов. `run_with_iobinding()` позволяет pre-allocate GPU tensors.

```python
# Текущий:
outputs = self._session.run(self._output_names, {self._input_name: batch_tensor})

# С IO Binding:
ort_inputs = {
    self._input_name: ort.OrtValue.ortvalue_from_numpy(batch_tensor, 'cuda', 0)
}
ort_outputs = {
    name: None for name in self._output_names
}
binding = self._session.io_binding()
binding.bind_ortvalue_input(self._input_name, ort_inputs[self._input_name])
for name in self._output_names:
    binding.bind_output(name, 'cuda', 0)
self._session.run_with_iobinding(binding)
```

**Ожидаемый выигрыш:** 10-20% (eliminates CPU↔GPU copies). Benchmark needed.

**Сложность:** 4-6 часов. Requires careful memory management.

---

### 4.5. SSE timeout и graceful shutdown

**Проблема 1:** `process.py:114-144` SSE stream висит бесконечно если worker умирает. Нет terminal timeout.

```python
# process.py:131 — бесконечный pubsub.listen():
async for message in pubsub.listen():
    if message["type"] == "message":
        yield f"data: {message['data'].decode()}\n\n"
        # Нет timeout!
```

**Решение:** Добавить `asyncio.wait_for` с timeout (60s) + fallback poll.

**Проблема 2:** `worker.py:167-168` shutdown handler пустой.

```python
async def shutdown(ctx: dict[str, Any]) -> None:
    logger.info("Video processing worker shutting down")
    # Нужно: engine.dispose(), cleanup
```

**Файлы для изменения:**
- `backend/app/routes/process.py:114-144` — добавить timeout
- `ml/src/worker.py:167-168` — добавить cleanup

**Сложность:** 3-4 часа.

---

### 4.6. Thread-safe worker URL cache

**Проблема:** `vastai/client.py:27-29` — global mutable `_worker_url_cache` без thread-safety. При concurrent requests возможна race condition на read-modify-write.

```python
# client.py:27-29:
_worker_url_cache: str | None = None      # global mutable
_worker_url_cache_time: float = 0.0       # global mutable
```

**Решение:** `threading.Lock` вокруг cache access.

**Сложность:** 30 минут.

---

## 5. Tier 3: Strategic Investments (1-2 weeks)

### 5.1. FP16 quantization RTMO

**Описание:** Конвертировать RTMO-M ONNX model в FP16 offline. Tensor cores на Ampere (RTX 3050 Ti = GA107) получают 1.5-2x speedup на FP16 матрицах.

**Ожидаемый выигрыш:**
- 2x VRAM savings на activations → batch_size=16 возможен
- 1.5-2x speedup на tensor core ops
- <0.1 AP accuracy loss (по данным ONNX runtime documentation)

**Инструмент:** `onnxruntime.quantization.quantize_dynamic()` или `onnxruntime.quantization.quantize_static()`.

```bash
python -m onnxruntime.quantization.preprocess --input rtmo-m.onnx --output rtmo-m-preprocessed.onnx
python -m onnxruntime.quantization.quantize_static \
  --model_input rtmo-m-preprocessed.onnx \
  --model_output rtmo-m-fp16.onnx \
  --calibrate_datadir ./calibration_data \
  --quant_format QDQ
```

**Комбинация с batch inference:**
| Конфигурация | Inference time (364 frames) | VRAM |
|-------------|----------------------------|------|
| Текущий (FP32, batch=1) | 5.6s | 712MB |
| Batch=8 (FP32) | ~2.2s | ~937MB |
| Batch=16 (FP32) | ~1.4s | ~1.6GB |
| Batch=16 (FP16) | ~0.8-1.0s | ~800MB |

**Сложность:** 1-2 дня (quantization + validation + benchmark). Risk: accuracy regression needs A/B test.

---

### 5.2. CUDA Graph Capture

**Описание:** Для fixed `batch_size=8` shape, CUDA Graph capture устраняет kernel launch overhead (5-15% speedup).

**Ограничение:** Работает только с fixed input shapes. Dynamic shapes (variable video resolution) требуют re-capture.

**Сложность:** 3-5 дней. Требует深入了解 ONNX Runtime CUDA Graph API.

---

### 5.3. NVENC/NVDEC hardware acceleration

**Описание:** Заменить CPU encode/decode на GPU:

| Операция | Текущий | Замена | Speedup |
|----------|---------|--------|---------|
| Decode | `cv2.VideoCapture` (CPU) | `hwaccel=cuda` | 8x |
| Encode | `libx264` (CPU) | `h264_nvenc` (GPU) | 10x |

**Файлы:**
- `ml/src/utils/video_writer.py:30` — `codec="libx264"` → `codec="h264_nvenc"`
- `ml/src/utils/frame_buffer.py:61` — добавить `hwaccel=cuda` в `cv2.VideoCapture`

**Риски:**
- GPU contention: NVENC + RTMO inference на одной GPU (RTX 3050 Ti)
- Поддержка NVDEC через cv2/PyAV нестабильна на Linux
- `frame_buffer.py:61` использует `cv2.VideoCapture` — NVDEC через OpenCV требует сборки с `WITH_CUDA=ON`

**Сложность:** 2-3 дня (validation + fallback logic). Benchmark needed для оценки contention.

---

### 5.4. Physics + Recommendations parallel

**Проблема:** В `pipeline.py:307-336` physics calculations и recommendations выполняются последовательно, но recommendations зависят только от `metrics` (не от physics).

```python
# pipeline.py:307-336 — текущий:
physics_dict = compute_physics(poses_3d, phases)  # 307-329
recommendations = recommender.recommend(metrics, element_type)  # 334
```

**Решение:** Запустить physics и recommendations параллельно через `asyncio.gather` (в `analyze_async`). Но это micro-optimization (~10-50ms total).

**Сложность:** 30 минут (в `analyze_async` path).

---

### 5.5. Post-process task decomposition (arq `_depends_on`)

**Описание:** Разделить monolithic `process_video_task` на GPU task + CPU post-processing task через arq `_depends_on`. Это позволит запустить следующий GPU job параллельно с CPU post-processing текущего.

```
GPU task (process_video_task)          CPU task (post_process_task)
    │                                       │
    ├── RTMO inference                      ├── Metrics computation
    ├── Tracking                            ├── Recommendations
    ├── Gap filling                         ├── DB save
    ├── 3D lift                             └── PR check
    └── Upload to R2                            │
         │                                      │
         └── depends_on ────────────────────────┘
```

**Сложность:** 3-5 дней. Требует careful error handling и data transfer через R2.

---

## 6. Bug Fixes

### 6.1. Критические: Sync-in-async (блокирует event loop)

| # | Endpoint | Файл:строка | Вызов | Влияние |
|---|----------|-------------|-------|---------|
| 1 | `POST /detect` | `backend/app/routes/detect.py:39` | `upload_bytes()` (sync boto3) | Блокирует event loop на ~100ms |
| 2 | `POST /uploads/init` | `backend/app/routes/uploads.py:27,31,43` | `_client()` (sync boto3) | Блокирует на ~100ms |
| 3 | `POST /uploads/complete` | `backend/app/routes/uploads.py:73,84` | `_client()` (sync boto3) | Блокирует на ~150ms |
| 4 | `GET /sessions/*` | `backend/app/routes/sessions.py:35,39-41` | `get_object_url()` (sync boto3) | Блокирует на ~50ms per URL |

### 6.2. Medium: Resource leaks и inefficiency

| # | Проблема | Файл:строка | Влияние |
|---|----------|-------------|---------|
| 5 | Valkey connect/close на каждый вызов | `backend/app/task_manager.py:29-37` (все функции) | 10+ unnecessary RTT per task |
| 6 | arq pool create/close на каждый enqueue | `backend/app/routes/detect.py:50-67`, `process.py:53-76` | Unnecessary connection overhead |
| 7 | httpx client не переиспользуется | `ml/src/vastai/client.py:133,186` | TLS handshake overhead |
| 8 | Thread-unsafe global cache | `ml/src/vastai/client.py:27-29` | Race condition при concurrent requests |
| 9 | N+1 DB queries в session saver | `backend/app/services/session_saver.py:51-56` | 12 queries вместо 1 batch |
| 10 | Empty shutdown handler | `ml/src/worker.py:167-168` | Resource leak on worker stop |

### 6.3. Low: Код качество

| # | Проблема | Файл:строка | Влияние |
|---|----------|-------------|---------|
| 11 | Дублированный sync/async pipeline | `ml/src/pipeline.py:181` vs `610` | Maintenance burden |
| 12 | SSE no timeout | `backend/app/routes/process.py:114-144` | Stream hangs forever |
| 13 | Error retry по string matching | `ml/src/worker.py:314-316` | Fragile, no HTTP status handling |

---

## 7. Implementation Order

```
Неделя 1: Bug fixes + connection pooling
═══════════════════════════════════════

День 1-2:
  [3.1] Sync-in-async fixes (detect.py, uploads.py, sessions.py)  ← 3 часа
  [6.6] Thread-safe worker URL cache                               ← 30 мин
  [3.6] httpx.AsyncClient reuse                                    ← 1 час

День 2-3:
  [3.2] Valkey connection pool singleton                           ← 3-4 часа
  [3.3] arq pool singleton                                         ← 1 час
  [3.5] N+1 batch query в session_saver                            ← 2-3 часа
  [4.5] SSE timeout + worker graceful shutdown                     ← 3-4 часа

День 3-4:
  [4.6] Double buffering (AsyncFrameReader → pose_extractor)       ← 4-6 часов
  [4.3] ONNX SessionOptions optimization                           ← 2-3 часа

  Итого неделя 1: ~15-20 часов работы
  Ожидаемый cumulative speedup: ~15% (connection pooling + double buffering)

Неделя 2: GPU optimizations
══════════════════════════

День 5-7:
  [4.1] Batch RTMO inference integration                          ← 2-3 дня
  [4.4] ONNX IO Binding (zero-copy)                               ← 4-6 часов

  Итого неделя 2: ~20-25 часов
  Ожидаемый cumulative speedup: ~2.5-3x (batch inference dominant)

Неделя 3+: Strategic (optional)
══════════════════════════════

  [5.1] FP16 quantization                                         ← 1-2 дня
  [5.3] NVENC/NVDEC hardware acceleration                          ← 2-3 дня
  [5.2] CUDA Graph Capture                                        ← 3-5 дней
  [5.5] Post-process task decomposition                            ← 3-5 дней
```

### Dependency graph

```
[3.1] sync-in-async ─────────────────────────────────────────────┐
[3.2] Valkey pool ──── [3.3] arq pool ───────────────────────────┤
[3.5] N+1 batch query ───────────────────────────────────────────┤→ Testing
[3.6] httpx reuse ───────────────────────────────────────────────┤
[6.6] Thread-safe cache ─────────────────────────────────────────┘
    │
    ▼
[4.5] SSE timeout + shutdown ───────────────────────────────────→ Testing
    │
    ▼
[4.6] Double buffering ── [4.3] SessionOptions ─────────────────→ Benchmark
    │
    ▼
[4.1] Batch RTMO ──── [4.4] IO Binding ────────────────────────→ Benchmark
    │
    ▼
[5.1] FP16 ── [5.2] CUDA Graph ── [5.3] NVENC ── [5.5] Tasks → Benchmark
```

---

## 8. Risk Assessment

| Риск | Вероятность | Влияние | Mitigation |
|------|-------------|---------|------------|
| **Batch inference ломает tracking** | Средняя | Высокое | BatchPoseExtractor уже существует с упрощённым tracking. Интегрировать постепенно: batch inference → sequential tracking post-process. |
| **FP16 degrade accuracy** | Низкая | Среднее | Validate на skating dataset (<0.1 AP expected). Fallback на FP32. |
| **NVENC + RTMO GPU contention** | Высокая | Среднее | Benchmark contention first. Priority: NVENC только если decode/encode > 20% total time. |
| **Connection pooling memory leak** | Низкая | Среднее | FastAPI lifespan cleanup. Monitor с `redis-cli info clients`. |
| **ONNX IO Binding segfault** | Низкая | Высокое | ONNX Runtime stable API, но requires correct device placement. Comprehensive test suite. |
| **SSE timeout слишком агрессивный** | Средняя | Низкое | Configurable timeout с fallback на polling. Default 60s. |
| **raced condition на Vast.ai cache** | Низкая | Низкое | threading.Lock, worst case = дублированный route request. |
| **Batch size не optimal для 4K video** | Средняя | Низкое | Auto-tune batch_size по доступной VRAM и input resolution. |

---

## Приложение A: Files Index

| Файл | Строки | Роль |
|------|--------|------|
| `backend/app/storage.py` | 145 | R2/S3 client (sync + async) |
| `backend/app/task_manager.py` | 216 | Valkey task state (connect-per-call) |
| `backend/app/routes/detect.py` | 117 | POST /detect endpoint |
| `backend/app/routes/process.py` | 145 | POST /process endpoint |
| `backend/app/routes/uploads.py` | 92 | Chunked upload endpoints |
| `backend/app/routes/sessions.py` | 147 | Session CRUD |
| `backend/app/services/session_saver.py` | 82 | Save metrics to Postgres |
| `ml/src/worker.py` | 467 | arq worker (process_video_task, detect_video_task) |
| `ml/src/pipeline.py` | 876 | AnalysisPipeline (sync + async) |
| `ml/src/pose_estimation/pose_extractor.py` | ~950 | Main RTMO extractor (per-frame) |
| `ml/src/pose_estimation/rtmo_batch.py` | 282 | BatchRTMO (NOT integrated) |
| `ml/src/pose_estimation/batch_extractor.py` | 358 | BatchPoseExtractor (NOT integrated) |
| `ml/src/utils/frame_buffer.py` | 92 | AsyncFrameReader (NOT used in pipeline) |
| `ml/src/utils/video_writer.py` | 60 | H264Writer (libx264, not NVENC) |
| `ml/src/vastai/client.py` | 205 | Vast.ai Serverless dispatch |
| `ml/src/visualization/comparison.py` | ~250 | Comparison renderer |

---

## Приложение B: Expected Cumulative Speedup

| Этап | Total time (364 frames) | Speedup vs baseline |
|------|------------------------|---------------------|
| Baseline (текущий) | ~9.3s | 1.0x |
| + Connection pooling (Tier 1) | ~9.3s | 1.0x (no change in pipeline, only I/O) |
| + Double buffering (Tier 2) | ~8.6s | 1.08x |
| + ONNX SessionOptions (Tier 2) | ~7.7s | 1.21x |
| + Batch RTMO batch=8 (Tier 2) | ~4.3s | 2.16x |
| + IO Binding (Tier 2) | ~3.6s | 2.58x |
| + FP16 (Tier 3) | ~2.0-2.5s | 3.7-4.6x |
| + NVENC encode (Tier 3) | depends on encode fraction | needs benchmark |

> **Примечание:** Inference (5.6s) — 60% текущего pipeline time. Batch inference — single biggest win. Все остальные оптимизации дают <20% cumulative.
