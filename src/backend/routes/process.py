"""POST /api/process — run the analysis pipeline with SSE progress."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from src.backend.schemas import (
    ProcessRequest,
    ProcessResponse,
    ProcessStats,
    QueueProcessResponse,
    TaskStatusResponse,
)
from src.config import get_settings
from src.task_manager import (
    create_task_state,
    get_task_state,
    get_valkey_client,
    set_cancel_signal,
)
from src.types import PersonClick
from src.web_helpers import PipelineCancelled, process_video_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()

OUTPUTS_DIR = Path("data/uploads")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

_executor = ThreadPoolExecutor(max_workers=1)

# Shared cancel event — one concurrent pipeline at a time
_cancel_event = threading.Event()


@router.post("/api/process")
async def process_video(req: ProcessRequest) -> EventSourceResponse:
    """Run the visualization pipeline and stream progress via SSE."""

    async def event_generator():
        progress_queue: asyncio.Queue = asyncio.Queue()
        done_event = asyncio.Event()

        _cancel_event.clear()

        def progress_cb(fraction: float, message: str) -> None:
            progress_queue.put_nowait({"progress": round(fraction, 3), "message": message})

        def run_pipeline():
            try:
                click = PersonClick(x=req.person_click.x, y=req.person_click.y)
                src_name = Path(req.video_path).stem
                out_path = OUTPUTS_DIR / f"{src_name}_analyzed.mp4"

                result = process_video_pipeline(
                    video_path=req.video_path,
                    person_click=click,
                    frame_skip=req.frame_skip,
                    layer=req.layer,
                    tracking=req.tracking,
                    blade_3d=False,
                    export=req.export,
                    output_path=str(out_path),
                    progress_cb=progress_cb,
                    cancel_event=_cancel_event,
                    depth=req.depth,
                    optical_flow=req.optical_flow,
                    segment=req.segment,
                    foot_track=req.foot_track,
                    matting=req.matting,
                    inpainting=req.inpainting,
                )

                stats = result["stats"]
                video_rel = (
                    str(out_path.relative_to(OUTPUTS_DIR))
                    if out_path.is_relative_to(OUTPUTS_DIR)
                    else out_path.name
                )
                poses_rel = None
                csv_rel = None
                if result["poses_path"]:
                    pp = Path(result["poses_path"])
                    poses_rel = (
                        str(pp.relative_to(OUTPUTS_DIR))
                        if pp.is_relative_to(OUTPUTS_DIR)
                        else pp.name
                    )
                if result["csv_path"]:
                    cp = Path(result["csv_path"])
                    csv_rel = (
                        str(cp.relative_to(OUTPUTS_DIR))
                        if cp.is_relative_to(OUTPUTS_DIR)
                        else cp.name
                    )

                response = ProcessResponse(
                    video_path=video_rel,
                    poses_path=poses_rel,
                    csv_path=csv_rel,
                    stats=ProcessStats(
                        total_frames=stats["total_frames"],
                        valid_frames=stats["valid_frames"],
                        fps=stats["fps"],
                        resolution=stats["resolution"],
                    ),
                    status="Анализ завершён!",
                )
                progress_queue.put_nowait({"event": "result", "data": response.model_dump_json()})
            except PipelineCancelled:
                progress_queue.put_nowait(
                    {"event": "cancelled", "data": json.dumps({"message": "Обработка отменена"})}
                )
            except Exception as e:
                logger.exception("Pipeline error")
                progress_queue.put_nowait({"event": "error", "data": json.dumps({"error": str(e)})})
            finally:
                done_event.set()

        # Run pipeline in background thread
        loop = asyncio.get_event_loop()
        loop.run_in_executor(_executor, run_pipeline)

        # Stream progress events as they arrive
        while not done_event.is_set() or not progress_queue.empty():
            try:
                evt = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
            except TimeoutError:
                continue

            if "event" in evt:
                yield {"event": evt["event"], "data": evt["data"]}
            else:
                yield {"event": "progress", "data": json.dumps(evt)}

    return EventSourceResponse(event_generator())


@router.post("/api/process/cancel")
async def cancel_processing():
    """Cancel the currently running pipeline."""
    _cancel_event.set()
    return {"status": "cancelled"}


@router.post("/api/process/queue", response_model=QueueProcessResponse)
async def enqueue_process(req: ProcessRequest):
    """Enqueue video processing job and return task_id immediately."""
    settings = get_settings()
    task_id = f"proc_{uuid.uuid4().hex[:12]}"

    valkey = await get_valkey_client()
    try:
        await create_task_state(task_id, video_path=req.video_path, valkey=valkey)
    finally:
        await valkey.close()

    arq_pool = await create_pool(
        RedisSettings(
            host=settings.valkey_host,
            port=settings.valkey_port,
            database=settings.valkey_db,
            password=settings.valkey_password,
        )
    )
    try:
        await arq_pool.enqueue_job(
            "process_video_task",
            task_id=task_id,
            video_path=req.video_path,
            person_click={"x": req.person_click.x, "y": req.person_click.y},
            frame_skip=req.frame_skip,
            layer=req.layer,
            tracking=req.tracking,
            export=req.export,
            depth=req.depth,
            optical_flow=req.optical_flow,
            segment=req.segment,
            foot_track=req.foot_track,
            matting=req.matting,
            inpainting=req.inpainting,
        )
    finally:
        await arq_pool.close()

    return QueueProcessResponse(task_id=task_id)


@router.get("/api/process/{task_id}/status", response_model=TaskStatusResponse)
async def get_process_status(task_id: str):
    """Poll task status."""
    valkey = await get_valkey_client()
    try:
        state = await get_task_state(task_id, valkey=valkey)
    finally:
        await valkey.close()

    if state is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Task not found")

    result = None
    if state.get("result"):
        result = ProcessResponse(**state["result"])

    return TaskStatusResponse(
        task_id=task_id,
        status=state["status"],
        progress=state["progress"],
        message=state.get("message", ""),
        result=result,
        error=state.get("error"),
    )


@router.post("/api/process/{task_id}/cancel")
async def cancel_queued_process(task_id: str):
    """Cancel a queued or running task via Valkey signal."""
    await set_cancel_signal(task_id)
    return {"status": "cancel_requested", "task_id": task_id}
