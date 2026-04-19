# Аудит параллелизма и async-оптимизаций

> **Дата:** 2026-04-19
> **Статус:** Design Document (аудит)
> **Связанные:** `docs/specs/2026-04-19-pipeline-parallelism-design.md` (оригинальный документ)
> **Источник:** Синтез из 5 специализированных агентов + верификация по исходному коду

---

## 1. Executive Summary

Оригинальный документ `pipeline-parallelism-design.md` сфокусирован на ML-pipeline оптимизациях (BatchRTMO, double buffering, FP16) и корректно идентифицирует их как основные точки роста. Однако аудит обнаружил **4 критических бага**, отсутствующих в оригинальном документе, а также ряд некорректных оценок.

**Главные выводы:**

1. **4 критических бага** требуют немедленного исправления — они ломают функциональность (multipart upload), ломают пагинацию, возвращают пустые user names в connections, и создают race condition при сохранении результатов.
2. **N+1 presigned URLs** в `list_sessions` — самая большая I/O проблема: 40 последовательных TLS handshake к R2 на каждый запрос списка сессий. Оригинальный документ упоминает sync-in-async, но не замечает масштаб проблемы.
3. **BatchRTMO** (оригинальный Tier 2, item 4.1) остаётся главным выигрышем в pipeline throughput — 2.5-4x speedup на inference.
4. **CUDA Graph и IO Binding уже реализованы** в `rtmo_batch.py` — оригинальный документ оценивает их в 3-5 и 4-6 дней, но реальный объём работы составляет 1-3 часа (интеграция существующего кода).
5. **NVENC уже реализован** — `H264Writer` auto-detects `h264_nvenc`. Item 5.3 оригинального документа частично устарел.

**Что было упущено в оригинальном документе:**
- 4 критических бага (sections 2.1-2.4)
- aiobotocore client создаётся заново на каждый вызов (section 3.2)
- `list_by_user()` missing `selectinload(metrics)` (section 3.3)
- `serve_output` делает 2 R2 round-trip (HEAD + GET) вместо 1 (section 3.4)
- bcrypt блокирует event loop (section 3.5)
- Partial commit в `update_session_analysis` (section 2.4)
- arq не поддерживает `_depends_on` — item 5.5 оригинального документа неосуществим (section 6.1)
- PoseExtractor создаётся внутри `detect_video_task` на каждый запрос (section 4.6)

---

## 2. Critical Bugs

### 2.1. Route path mismatch: multipart upload сломан

**Симптом:** Upload завершается с ошибкой 404 или 405.

**Причина:** Frontend отправляет `POST /uploads/{upload_id}/complete`, а backend регистрирует `POST /uploads/complete` (без path parameter).

| Компонент | Файл:строка | URL |
|-----------|-------------|-----|
| Frontend | `frontend/src/lib/api/uploads.ts:76` | `POST /uploads/${init.upload_id}/complete` |
| Backend | `backend/app/routes/uploads.py:70` | `@router.post("/uploads/complete")` |

**Фикс:** Добавить `upload_id` как path parameter в backend route:
```python
@router.post("/uploads/{upload_id}/complete")
```
Или исправить frontend на `POST /uploads/complete` с `upload_id` в body (менее RESTful, но проще).

**Приоритет:** P0 — функциональность полностью сломана.

---

### 2.2. Pagination bug: `total` всегда равно количеству элементов на странице

**Симптом:** Frontend pagination не работает — показывает "no more items" или неправильный total count.

**Причина:** `total=len(sessions)` возвращает количество элементов в текущей странице (до `limit`), а не общее количество записей.

| Endpoint | Файл:строка | Проблема |
|----------|-------------|----------|
| `GET /sessions` | `backend/app/routes/sessions.py:102` | `total=len(sessions)` — возвращает len(paginated result) |
| `GET /choreography/programs` | `backend/app/routes/choreography.py:258` | `total=len(programs)` — аналогично |

**Фикс:** Добавить `SELECT COUNT(*) ...` запрос в `list_by_user()` и `list_programs_by_user()`, вернуть `(results, total_count)` tuple.

```python
# crud/session.py — list_by_user должен возвращать (list, int)
count_q = select(func.count()).select_from(query.subquery())
total = (await db.execute(count_q)).scalar() or 0
```

**Приоритет:** P0 — пагинация сломана для всех list endpoints.

---

### 2.3. Connection user names всегда null

**Симптом:** В UI connections показывают пустые имена пользователей.

**Причина:** `ConnectionResponse` содержит `from_user_name` и `to_user_name` (строки 327-328 в `schemas.py`), но `Connection` model (`models/connection.py`) не имеет relationship к `User`. `_conn_to_response()` (`connections.py:36-37`) использует `model_validate(conn)` — Pydantic берёт `None` для отсутствующих атрибутов.

| Компонент | Файл:строка | Проблема |
|-----------|-------------|----------|
| Schema | `backend/app/schemas.py:327-328` | `from_user_name: str \| None = None`, `to_user_name: str \| None = None` |
| Model | `backend/app/models/connection.py:31-61` | Нет relationship к User |
| Route | `backend/app/routes/connections.py:36-37` | `_conn_to_response` не загружает имена |

**Фикс:** Вариант A — добавить SQLAlchemy relationship и `selectinload` в запрос. Вариант B — загружать имена явно в route handler через отдельный batch query.

**Приоритет:** P1 — функциональность деградирована, но не сломана полностью.

---

### 2.4. Partial commit: double commit в worker

**Симптом:** При сбое после `update_session_analysis()` но до `save_analysis_results()` session остаётся в статусе "completed" без metrics. При повторной обработке — stale data.

**Причина:** `update_session_analysis()` (`crud/session.py:101`) вызывает `await db.commit()` внутри CRUD функции. Затем worker (`worker.py:335`) вызывает `await db.commit()` снова. Если `save_analysis_results()` падает между двумя commit — данные частично сохранены.

| Компонент | Файл:строка | Проблема |
|-----------|-------------|----------|
| CRUD | `backend/app/crud/session.py:101` | `await db.commit()` внутри CRUD функции |
| Worker | `backend/app/worker.py:319-335` | Два последовательных commit в одной транзакции |

**Фикс:** Убрать `db.commit()` из `update_session_analysis()`. Пусть worker управляет транзакцией целиком через FastAPI dependency `get_db()` (который делает auto-commit).

```python
# crud/session.py:101 — убрать:
# await db.commit()  # ← УДАЛИТЬ
# Пусть caller решает когда коммитить
```

**Приоритет:** P0 — data consistency. Может привести к silent data loss.

---

## 3. Tier 1: Quick Wins (1-2 дня, ~12 часов)

### 3.1. N+1 presigned URLs: asyncio.gather + client pooling

**Проблема:** `list_sessions` (`sessions.py:101-102`) вызывает `_session_to_response()` для каждой session в цикле. Каждый вызов генерирует 2 presigned URL (video + processed_video). При `limit=20` = 40 последовательных `get_object_url_async()` вызовов, каждый из которых создаёт новый aiobotocore client с TLS handshake.

**Масштаб проблемы:** При `limit=20`:
- 40 async client creations (каждый = TLS handshake + connection setup)
- Каждый вызов ~50ms (presigned URL generation minimal, overhead в client creation)
- Total: ~2s на генерацию presigned URL для списка сессий

**Файлы:**
- `backend/app/routes/sessions.py:101-102` — sequential loop
- `backend/app/storage.py:106-116` — `_async_client()` создаёт client на каждый вызов

**Фикс:**

1. Singleton aiobotocore client в lifespan:
```python
# storage.py — module-level singleton
_s3_client: aiobotocore.AioClientCreator | None = None

async def init_s3_client():
    global _s3_client
    s = get_settings()
    _s3_client = _async_session.create_client(
        "s3",
        endpoint_url=s.r2.endpoint_url or None,
        aws_access_key_id=s.r2.access_key_id.get_secret_value(),
        aws_secret_access_key=s.r2.secret_access_key.get_secret_value(),
        config=BotoConfig(signature_version="s3v4"),
        region_name="auto",
    )
    # Warm up connection
    await _s3_client.__aenter__()

async def close_s3_client():
    global _s3_client
    if _s3_client:
        await _s3_client.__aexit__(None, None, None)
        _s3_client = None
```

2. `asyncio.gather` для presigned URLs:
```python
# sessions.py — parallel presigned URLs
async def _sessions_to_responses(sessions) -> list[SessionResponse]:
    tasks = [_session_to_response(s) for s in sessions]
    return await asyncio.gather(*tasks)
```

3. Параллельная генерация URL внутри `_session_to_response`:
```python
video_url, processed_video_url = await asyncio.gather(
    get_object_url_async(session.video_key) if session.video_key else asyncio.coroutine(lambda: session.video_url)(),
    get_object_url_async(session.processed_video_key) if session.processed_video_key else asyncio.coroutine(lambda: session.processed_video_url)(),
)
```

**Ожидаемый выигрыш:** 40 sequential → 2 batch (20 video + 20 processed). При singleton client: ~2s → ~50ms.

**Время:** 3-4 часа.

---

### 3.2. `list_by_user()` missing `selectinload(metrics)`

**Проблема:** `get_by_id()` (`crud/session.py:26-29`) загружает `metrics` через `selectinload(Session.metrics)`, но `list_by_user()` (`crud/session.py:32-50`) — нет. При обращении к `session.metrics` в response serialization — lazy load N+1.

**Файл:** `backend/app/crud/session.py:32-50`

**Фикс:** Добавить `.options(selectinload(Session.metrics))` в `list_by_user()`.

```python
query = select(Session).options(selectinload(Session.metrics)).where(Session.user_id == user_id)
```

**Время:** 15 минут.

---

### 3.3. `serve_output`: 2 R2 round-trips вместо 1

**Проблема:** `serve_output` (`misc.py:29-45`) сначала вызывает `object_exists_async(key)` (HEAD request), затем `stream_object_async(key)` (GET request). Это 2 R2 round-trips.

**Файл:** `backend/app/routes/misc.py:29-45`

**Фикс:** Попытаться GET напрямую, обработать 404:
```python
@router.get("/outputs/{key:path}")
async def serve_output(key: str):
    try:
        body, length, ctype = await stream_object_async(key)
    except ClientError as e:
        if e.response["Error"]["Code"] == "404" or e.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404:
            raise HTTPException(status_code=404, detail="File not found")
        raise
    # ... rest
```

**Ожидаемый выигрыш:** 1 R2 round-trip вместо 2 (~50-100ms saved per request).

**Время:** 30 минут.

---

### 3.4. bcrypt блокирует event loop

**Проблема:** `hash_password()` и `verify_password()` (`auth/security.py:16-26`) используют `passlib` с bcrypt (rounds=12). bcrypt — CPU-bound операция, ~300ms на registration, ~300ms на login. В async handler (`auth.py:57,67`) это блокирует event loop на 300ms.

**Файлы:**
- `backend/app/auth/security.py:13` — `bcrypt__rounds=12`
- `backend/app/routes/auth.py:57` — `hash_password(body.password)` в `async def register`
- `backend/app/routes/auth.py:67` — `verify_password(body.password, ...)` в `async def login`

**Фикс:** Обернуть в `asyncio.to_thread`:
```python
# auth.py:57
hashed = await asyncio.to_thread(hash_password, body.password)
# auth.py:67
valid = await asyncio.to_thread(verify_password, body.password, user.hashed_password)
```

**Время:** 15 минут.

---

### 3.5. Sync-in-async в uploads routes

**Проблема:** `uploads.py:27,73` использует синхронный `_client()` (boto3) внутри `async def` handlers. Блокирует event loop на ~100-150ms.

**Файл:** `backend/app/routes/uploads.py:27,73`

**Фикс:** Заменить на async версии из `storage.py`. Async функции уже существуют (`upload_file_async`, etc.).

**Время:** 1 час (уже описано в оригинальном документе, item 3.1).

---

### 3.6. aiobotocore client pooling (storage.py)

**Проблема:** Каждая async функция в `storage.py` создаёт новый client через `_async_client()` (строки 106-116). Client creation = TLS handshake + connection setup.

**Файл:** `backend/app/storage.py:106-116`

**Фикс:** Singleton client в lifespan (см. item 3.1). Все async функции используют singleton вместо `async with await _async_client() as s3`.

**Время:** 2 часа (совместно с 3.1).

---

### 3.7. Valkey connection pool singleton

**Проблема:** `task_manager.py` создаёт и уничтожает `aioredis.Redis` соединение на каждый вызов. Для `process_video_task` — ~10+ connect/close циклов.

**Файл:** `backend/app/task_manager.py` (все функции)

**Фикс:** Singleton pool в lifespan (как описано в оригинальном документе, item 3.2).

**Время:** 3-4 часа (совместно с оригинальным item 3.2).

---

### 3.8. arq pool singleton

**Проблема:** `detect.py` и `process.py` создают/закрывают `arq.create_pool()` на каждый enqueue request.

**Файлы:** `backend/app/routes/detect.py`, `backend/app/routes/process.py`

**Фикс:** Singleton pool в lifespan (оригинальный документ, item 3.3).

**Время:** 1 час.

---

### 3.9. N+1 batch query в session_saver

**Проблема:** `session_saver.py:51-56` вызывает `get_current_best()` в цикле по каждому metric (12 queries).

**Файл:** `backend/app/services/session_saver.py:41-70`

**Фикс:** Batch query (оригинальный документ, item 3.5).

**Время:** 2-3 часа.

---

### 3.10. `total=len()` pagination fix

**Проблема:** Описано в section 2.2.

**Файлы:** `backend/app/routes/sessions.py:102`, `backend/app/routes/choreography.py:258`

**Фикс:** COUNT query в CRUD layer.

**Время:** 1-2 часа.

---

## 4. Tier 2: Medium-Term (3-7 дней, ~25 часов)

### 4.1. Batch RTMO inference integration

**Проблема:** `PoseExtractor.extract_video_tracked()` вызывает rtmlib per-frame (batch=1). `BatchRTMO` (`rtmo_batch.py:175`) и `BatchPoseExtractor` (`batch_extractor.py:44`) существуют, но не подключены к pipeline.

**Текущее состояние BatchPoseExtractor:**
- Использует `BatchRTMO` напрямую (обходит rtmlib)
- Отсутствует tracking (comment: `target_track_id=None, # Tracking to be implemented`)
- Работает без Sports2D/DeepSORT trackers

**Профиль VRAM (batch_size=8):**

| Компонент | VRAM |
|-----------|------|
| RTMO-M model weights | ~100MB |
| Batch input (8 frames x 640x640x3) | ~37MB |
| Batch activations | ~800MB (est.) |
| **Total est.** | **~937MB** |

VRAM headroom на RTX 3050 Ti (4GB): ~3GB. batch_size=16 возможно (~1.6GB activations).

**Файлы:**
- `ml/src/pose_estimation/pose_extractor.py:238-300` — заменить per-frame loop на batched
- `ml/src/pose_estimation/batch_extractor.py:44-315` — добавить tracking support

**Ожидаемый выигрыш:** 2.5-4x на inference. Текущий 5.6s → ~1.4-2.2s для 364 кадров.

**Время:** 2-3 дня (как в оригинальном документе, item 4.1).

---

### 4.2. Double buffering (AsyncFrameReader)

**Проблема:** `AsyncFrameReader` существует в `frame_buffer.py:22`, используется в `web_helpers.py:320-323`, но НЕ используется в `pose_extractor.py:255`.

**Файлы:**
- `ml/src/utils/frame_buffer.py:22` — AsyncFrameReader
- `ml/src/pose_estimation/pose_extractor.py:238-300` — заменить `cap.read()` на `AsyncFrameReader`

**Ожидаемый выигрыш:** ~8ms/frame * 364 frames = ~2.9s. Но при batch inference (4.1) overlap частичный — net savings ~0.7s.

**Время:** 4-6 часов (оригинальный документ, item 4.2).

---

### 4.3. ONNX SessionOptions optimization

**Проблема:** `BatchRTMO.__init__()` (`rtmo_batch.py:229`) создаёт `InferenceSession` без `SessionOptions`.

**Файл:** `ml/src/pose_estimation/rtmo_batch.py:229`

**Фикс:** Добавить `graph_optimization_level=ORT_ENABLE_ALL`, `cudnn_conv_algo_search=EXHAUSTIVE`.

**Ожидаемый выигрыш:** 10-20% на inference.

**Время:** 2-3 часа (оригинальный документ, item 4.3).

---

### 4.4. IO Binding integration (уже реализовано)

**Важная коррекция:** Оригинальный документ оценивает IO Binding в 4-6 дней (item 4.4). Метод `infer_batch_iobinding()` уже реализован в `rtmo_batch.py:362`. Реальный объём работы — переключить `infer_batch()` на `infer_batch_iobinding()` в `BatchPoseExtractor` и провести benchmark.

**Файл:** `ml/src/pose_estimation/rtmo_batch.py:362`

**Время:** 2-3 часа (а не 4-6 дней).

---

### 4.5. CUDA Graph integration (уже реализовано)

**Важная коррекция:** Оригинальный документ оценивает CUDA Graph в 3-5 дней (item 5.2). Метод `_enable_cuda_graph()` уже реализован в `rtmo_batch.py:258-322`. Реальный объём работы — вызвать его после инициализации `BatchRTMO` с фиксированным `batch_size`.

**Файл:** `ml/src/pose_estimation/rtmo_batch.py:258-322`

**Ограничение:** Работает только с fixed input shapes. Требует re-capture при изменении batch_size.

**Время:** 1-2 часа (а не 3-5 дней).

---

### 4.6. PoseExtractor в worker startup

**Проблема:** `detect_video_task` (`worker.py:406-413`) создаёт `PoseExtractor` внутри функции на каждый запрос. Model loading занимает ~1-2s (cold start).

**Файл:** `backend/app/worker.py:406-413`

**Фикс:** Инициализировать `PoseExtractor` в `startup(ctx)` и сохранить в `ctx`:
```python
async def startup(ctx: dict[str, Any]) -> None:
    from src.pose_estimation.pose_extractor import PoseExtractor
    ctx["pose_extractor"] = PoseExtractor(mode="balanced", ...)
```

**Время:** 1 час.

---

### 4.7. Redundant CoM computation

**Проблема:** `calculate_center_of_mass` вызывается 3 раза в разных стадиях pipeline на одних и тех же данных.

**Файлы:** `ml/src/analysis/physics_engine.py:165-188`, `ml/src/analysis/metrics.py:467,793`

**Фикс:** Вычислить CoM один раз, передать через pipeline context.

**Время:** 1-2 часа.

---

### 4.8. `curve_fit` → `np.polyfit`

**Проблема:** `scipy.optimize.curve_fit` используется для parabolic fit в `physics_engine.py:412,486`. `np.polyfit` делает то же самое для degree=2, но ~50x быстрее (no iterative optimization).

**Файл:** `ml/src/analysis/physics_engine.py:412,486`

**Фикс:** Заменить `curve_fit(parabola, t, flight_com)` на `np.polyfit(t, flight_com, 2)`.

**Время:** 30 минут.

---

### 4.9. ComparisonRenderer parallel extraction

**Проблема:** `ComparisonRenderer` извлекает poses для двух видео последовательно (~11.2s total). Параллельное извлечение сократит вдвое.

**Файл:** `ml/src/visualization/comparison.py:192-224`

**Фикс:** `asyncio.gather` или `concurrent.futures.ThreadPoolExecutor` для двух extraction.

**Время:** 2-3 часа.

---

## 5. Tier 3: Strategic (1-2 недели)

### 5.1. FP16 quantization RTMO

**Описание:** Конвертировать RTMO-M ONNX model в FP16. Tensor cores на RTX 3050 Ti (Ampere GA107) получают 1.5-2x speedup на FP16.

**Комбинация с batch inference:**

| Конфигурация | Inference time (364 frames) | VRAM |
|-------------|----------------------------|------|
| Текущий (FP32, batch=1) | 5.6s | 712MB |
| Batch=8 (FP32) | ~2.2s | ~937MB |
| Batch=16 (FP32) | ~1.4s | ~1.6GB |
| Batch=16 (FP16) | ~0.8-1.0s | ~800MB |

**Время:** 1-2 дня (оригинальный документ, item 5.1).

---

### 5.2. Post-process task decomposition (ручное enqueue)

**Важная коррекция:** Оригинальный документ (item 5.5) предлагает использовать arq `_depends_on` для chaining GPU и CPU tasks. **arq не поддерживает `_depends_on`.** Это API не существует в arq.

**Альтернатива:** Ручной enqueue в конце GPU task:
```python
# В конце process_video_task:
await ctx["pool"].enqueue_job("post_process_task", session_id=session_id)
```

**Время:** 3-5 дней (как в оригинале, но с другой реализацией).

---

### 5.3. NVDEC hardware decode

**Важная коррекция:** NVENC encode уже реализован в `H264Writer` (`video_writer.py:60`). Оригинальный документ (item 5.3) предлагает добавить NVENC — это уже сделано.

Что осталось не реализовано: NVDEC decode (GPU-accelerated video decode).

**Файл:** `ml/src/utils/video_writer.py:60` — NVENC auto-detection.

**Время для NVDEC:** 2-3 дня (требует cv2 с CUDA backend или PyAV hwaccel).

---

### 5.4. Worker shutdown: GPU cleanup

**Проблема:** `shutdown()` (`worker.py:169-174`) только закрывает Valkey pool. ONNX sessions не освобождаются — VRAM leak на restart.

**Файл:** `backend/app/worker.py:169-174`

**Фикс:** Добавить cleanup для всех ONNX sessions и PoseExtractor:
```python
async def shutdown(ctx: dict[str, Any]) -> None:
    extractor = ctx.get("pose_extractor")
    if extractor:
        # Cleanup ONNX sessions, release VRAM
        del extractor
    pool = ctx.get("redis")
    if pool:
        await pool.close()
```

**Время:** 1-2 часа.

---

### 5.5. No checkpointing для Vast.ai tasks

**Проблема:** SIGTERM во время Vast.ai processing теряет результаты. Нет intermediate checkpoint.

**Файл:** `backend/app/worker.py:177-353`

**Время:** 2-3 дня (design + implementation).

---

### 5.6. Frontend: SSE вместо polling

**Проблема:** Session detail page (`sessions/[id]/page.tsx:19`) polls каждые 3s. SSE endpoint существует (`process.py:99-138`), но frontend его не использует.

**Файлы:**
- `frontend/src/app/(app)/sessions/[id]/page.tsx:19` — polling
- `backend/app/routes/process.py:99-138` — SSE endpoint

**Время:** 2-3 часа.

---

### 5.7. Thread pool oversubscription

**Проблема:** ONNX sessions имеют конфликтующие thread settings (`intra=2` vs `intra=1`). Нет `OMP_NUM_THREADS`.

**Файлы:** `ml/src/pose_estimation/rtmo_batch.py:232`, `ml/src/pose_estimation/onnx_extractor.py:53`

**Фикс:** Установить `OMP_NUM_THREADS=4` (или `intra_op_num_threads=4`) для ONNX Runtime, проверить что rtmlib использует те же настройки.

**Время:** 1-2 часа.

---

## 6. Design Doc Corrections

Следующие пункты оригинального документа `pipeline-parallelism-design.md` содержат ошибки или устарели:

### 6.1. arq `_depends_on` не существует (item 5.5)

**Оригинал:** "Разделить monolithic `process_video_task` на GPU task + CPU post-processing task через arq `_depends_on`."

**Реальность:** arq не имеет API `_depends_on`. Это не существует в arq. Для chaining задач необходимо ручной enqueue в конце первой задачи.

### 6.2. CUDA Graph уже реализован (item 5.2)

**Оригинал:** Оценка 3-5 дней, "Требует深入了解 ONNX Runtime CUDA Graph API."

**Реальность:** `_enable_cuda_graph()` уже реализован в `rtmo_batch.py:258-322`. Метод полностью готов к использованию. Реальный объём — 1-2 часа на вызов и benchmark.

### 6.3. IO Binding уже реализован (item 4.4)

**Оригинал:** Оценка 4-6 часов.

**Реальность:** `infer_batch_iobinding()` уже реализован в `rtmo_batch.py:362`. Нужно только переключить вызов. Реальный объём — 2-3 часа.

### 6.4. NVENC уже реализован (item 5.3, частично)

**Оригинал:** "Replace `libx264` (CPU) with `h264_nvenc` (GPU)" — оценка 2-3 дня.

**Реальность:** `H264Writer` уже auto-detects `h264_nvenc` (`video_writer.py:60`). Encode optimization выполнена. Остался только NVDEC decode — отдельная задача.

### 6.5. Worker shutdown не пустой

**Оригинал:** "worker.py:167-168 shutdown handler пустой."

**Реальность:** shutdown handler закрывает Valkey pool (`worker.py:171-174`), но не освобождает GPU resources. Исправленная формулировка: "shutdown handler закрывает Valkey pool, но не освобождает ONNX sessions и VRAM."

### 6.6. Оценки cumulative speedup

Оригинальный документ (Приложение B) даёт optimistic cumulative speedup. Реалистичная оценка:

| Этап | Total time (364 frames) | Примечание |
|------|------------------------|------------|
| Baseline (текущий) | ~9.3s | Измерено |
| + Bug fixes (Tier 1) | ~9.3s | No pipeline change, only I/O |
| + Double buffering | ~8.6s | 0.7s net savings (partial overlap with batch) |
| + ONNX SessionOptions | ~7.7s | 10-20% inference savings |
| + Batch RTMO batch=8 | ~4.3s | 2.5-4x inference speedup |
| + IO Binding | ~3.6s | Zero-copy GPU transfer |
| + CUDA Graph | ~3.0-3.3s | 5-15% on kernel launch |
| + FP16 | ~2.0-2.5s | 1.5-2x tensor core speedup |

---

## 7. Implementation Order

### Dependency graph

```
Неделя 1: Bug fixes + I/O parallelism (Tier 1)
═══════════════════════════════════════════════

  День 1 (4 часа):
    [2.1] Route path mismatch fix                     ← 30 мин
    [2.2] Pagination COUNT query                       ← 1-2 часа
    [2.4] Partial commit fix                           ← 30 мин
    [3.2] list_by_user selectinload                    ← 15 мин
    [3.4] bcrypt asyncio.to_thread                     ← 15 мин
    [3.3] serve_output 1 R2 round-trip                 ← 30 мин

  День 2 (5 часов):
    [3.1] N+1 presigned URLs + S3 client singleton     ← 3-4 часа
    [3.5] Sync-in-async uploads.py fix                 ← 1 час

  День 3 (4 часа):
    [3.7] Valkey connection pool singleton             ← 3-4 часа
    [3.8] arq pool singleton                           ← 1 час (совместно)

  Итого неделя 1: ~13 часов
  Результат: 4 бага исправлены, I/O throughput ↑↑

Неделя 2: ML Pipeline optimizations (Tier 2)
═══════════════════════════════════════════════

  День 4-5:
    [4.1] Batch RTMO inference integration             ← 2-3 дня
    [4.2] Double buffering (AsyncFrameReader)          ← 4-6 часов (параллельно)

  День 6:
    [4.3] ONNX SessionOptions                          ← 2-3 часа
    [4.4] IO Binding switch                            ← 2-3 часа (уже реализован)
    [4.5] CUDA Graph enable                            ← 1-2 часа (уже реализован)
    [4.6] PoseExtractor in worker startup              ← 1 час
    [4.7] Redundant CoM computation                    ← 1-2 часа
    [4.8] curve_fit → np.polyfit                       ← 30 мин

  Итого неделя 2: ~20-25 часов
  Результат: Pipeline 9.3s → ~3.6s (2.6x)

Неделя 3+: Strategic (Tier 3, optional)
═══════════════════════════════════

  [5.1] FP16 quantization                             ← 1-2 дня
  [5.2] Post-process task decomposition               ← 3-5 дней
  [5.3] NVDEC decode                                  ← 2-3 дня
  [5.4] Worker GPU cleanup                            ← 1-2 часа
  [5.5] Vast.ai checkpointing                         ← 2-3 дня
  [5.6] Frontend SSE                                  ← 2-3 часа
  [5.7] Thread pool oversubscription                  ← 1-2 часа
```

### Critical path

```
[2.1] ──┐
[2.2] ──┤
[2.4] ──┼──→ Testing (Tier 1 regression)
[3.2] ──┤
[3.4] ──┤
[3.3] ──┘
  │
  ▼
[3.1] N+1 presigned URLs ──→ Benchmark
  │
  ▼
[4.1] Batch RTMO ──→ Benchmark ←── [4.4] IO Binding
  │                          │
  ▼                          ▼
[4.5] CUDA Graph          [4.3] SessionOptions
  │
  ▼
[5.1] FP16 ──→ Final Benchmark
```

---

## 8. Risk Assessment

| Риск | Вероятность | Влияние | Mitigation |
|------|-------------|---------|------------|
| **Batch inference ломает tracking** | Средняя | Высокое | Batch inference → sequential tracking post-process. BatchPoseExtractor уже существует. |
| **FP16 degrade accuracy** | Низкая | Среднее | Validate на skating dataset. Fallback на FP32. |
| **S3 client singleton leak** | Низкая | Среднее | Lifespan cleanup + health check. |
| **Partial commit fix ломает existing flow** | Средняя | Высокое | `update_session_analysis` callers проверяют — единственный caller в worker.py. Тест на rollback scenario. |
| **Route path fix требует frontend redeploy** | Высокая | Низкое | Координировать backend + frontend deploy. |
| **Pagination COUNT query slow на больших таблицах** | Низкая | Низкое | Index на `(user_id, element_type)` уже есть. |
| **bcrypt rounds=12 слишком медленный под нагрузкой** | Низкая | Низкое | `asyncio.to_thread` снимает блокировку. При необходимости снизить до rounds=10. |
| **CUDA Graph re-capture overhead** | Средняя | Низкое | Только при изменении batch_size — редко в production. |
| **Connection model relationship migration** | Средняя | Среднее | Alembic migration + backward compatibility. |

---

## Приложение A: Files Index

| Файл | Строки | Роль | Критические проблемы |
|------|--------|------|---------------------|
| `backend/app/routes/sessions.py` | 145 | Session CRUD | N+1 URLs (101-102), pagination bug (102) |
| `backend/app/routes/uploads.py` | 92 | Chunked upload | Route mismatch (70 vs frontend:76), sync-in-async (27,73) |
| `backend/app/routes/connections.py` | 115 | Connection API | User names always null (36-37) |
| `backend/app/routes/misc.py` | 46 | File serving | 2 R2 round-trips (29-45) |
| `backend/app/routes/auth.py` | 94 | Auth endpoints | bcrypt blocks event loop (57,67) |
| `backend/app/routes/choreography.py` | 258+ | Choreography | Pagination bug (258) |
| `backend/app/storage.py` | 196 | R2/S3 client | Client per call (106-116) |
| `backend/app/crud/session.py` | 103 | Session CRUD | Missing selectinload (32-50), premature commit (101) |
| `backend/app/schemas.py` | 328+ | Pydantic schemas | ConnectionResponse names (327-328) |
| `backend/app/models/connection.py` | 61 | Connection model | No User relationship |
| `backend/app/worker.py` | 470+ | arq worker | PoseExtractor per-request (406), no GPU cleanup (169) |
| `backend/app/auth/security.py` | 51 | Password hashing | bcrypt rounds=12 |
| `backend/app/task_manager.py` | 216 | Valkey state | Connect-per-call pattern |
| `backend/app/database.py` | 35 | DB engine | pool_pre_ping not set |
| `ml/src/pose_estimation/rtmo_batch.py` | 370+ | Batch RTMO | CUDA Graph exists (258), IO Binding exists (362) — both unused |
| `ml/src/pose_estimation/pose_extractor.py` | ~950 | Main extractor | Per-frame (238-300), no double buffering |
| `ml/src/pose_estimation/batch_extractor.py` | 358 | Batch extractor | No tracking support |
| `ml/src/utils/video_writer.py` | 70 | H264 writer | NVENC already implemented (60) |
| `ml/src/analysis/physics_engine.py` | 500+ | Physics engine | curve_fit overkill (412,486), redundant CoM (165) |
| `ml/src/visualization/comparison.py` | ~250 | Comparison | Sequential extraction (192-224) |
| `frontend/src/lib/api/uploads.ts` | 83 | Upload client | Wrong URL path (76) |
| `frontend/src/lib/api/sessions.ts` | 110 | Session client | No bug (pagination handled by backend) |
| `frontend/src/app/(app)/sessions/[id]/page.tsx` | 19+ | Session detail | Polling instead of SSE (19) |
