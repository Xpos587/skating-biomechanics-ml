"""POST /api/process — run the analysis pipeline with SSE progress."""

from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from src.backend.schemas import ProcessRequest, ProcessResponse, ProcessStats
from src.types import PersonClick
from src.web_helpers import process_video_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()

OUTPUTS_DIR = Path("data/uploads")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

_executor = ThreadPoolExecutor(max_workers=1)


@router.post("/api/process")
async def process_video(req: ProcessRequest) -> EventSourceResponse:
    """Run the visualization pipeline and stream progress via SSE."""

    async def event_generator():
        progress_queue: asyncio.Queue = asyncio.Queue()
        done_event = asyncio.Event()

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
