"""arq worker for video processing pipeline.

Run with: uv run python -m skating_ml.worker

Dispatches all processing to Vast.ai Serverless GPU.
No local GPU fallback.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any, ClassVar

from arq import Retry
from arq.connections import RedisSettings

from backend.app.config import get_settings
from backend.app.task_manager import (
    TaskStatus,
    get_valkey_client,
    store_error,
    store_result,
)

logger = logging.getLogger(__name__)


async def startup(ctx: dict[str, Any]) -> None:
    logger.info("Video processing worker starting up")


async def shutdown(ctx: dict[str, Any]) -> None:
    logger.info("Video processing worker shutting down")


async def process_video_task(
    ctx: dict[str, Any],
    *,
    task_id: str,
    video_key: str,
    person_click: dict[str, int],
    frame_skip: int = 1,
    layer: int = 3,
    tracking: str = "auto",
    export: bool = True,
    ml_flags: dict[str, bool] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """arq task: dispatch video processing to Vast.ai Serverless GPU."""
    if ml_flags is None:
        ml_flags = {
            "depth": False,
            "optical_flow": False,
            "segment": False,
            "foot_track": False,
            "matting": False,
            "inpainting": False,
        }
    settings = get_settings()
    valkey = await get_valkey_client()

    try:
        now = datetime.now(UTC).isoformat()
        await valkey.hset(
            f"task:{task_id}",
            mapping={"status": TaskStatus.RUNNING, "started_at": now},
        )

        from backend.app.crud.session import get_by_id
        from backend.app.database import async_session
        from skating_ml.vastai.client import process_video_remote

        # Fetch element_type from session if session_id provided
        element_type = None
        if session_id:
            async with async_session() as db:
                session = await get_by_id(db, session_id)
                if session:
                    element_type = session.element_type

        logger.info("Dispatching task %s to Vast.ai (video_key=%s)", task_id, video_key)
        vast_result = await asyncio.to_thread(
            process_video_remote,
            video_key=video_key,
            person_click={"x": person_click["x"], "y": person_click["y"]},
            frame_skip=frame_skip,
            layer=layer,
            tracking=tracking,
            export=export,
            ml_flags=ml_flags,
            element_type=element_type,
        )
        logger.info("Vast.ai processing complete for task %s", task_id)

        response_data = {
            "video_path": vast_result.video_key,
            "poses_path": vast_result.poses_key,
            "csv_path": vast_result.csv_key,
            "stats": vast_result.stats,
            "status": "Analysis complete!",
        }
        await store_result(task_id, response_data, valkey=valkey)

        # Save analysis results to Postgres if session_id was provided
        if session_id and vast_result.metrics:
            try:
                from backend.app.database import async_session
                from backend.app.services.session_saver import save_analysis_results

                async with async_session() as db:
                    await save_analysis_results(
                        db,
                        session_id=session_id,
                        metrics=vast_result.metrics,
                        phases=vast_result.phases,
                        recommendations=vast_result.recommendations or [],
                    )
                    await db.commit()
            except Exception as save_err:
                logger.warning("Failed to save session results: %s", save_err)

        return response_data

    except Exception as e:
        logger.exception("Pipeline task %s failed", task_id)
        await store_error(task_id, str(e), valkey=valkey)
        error_msg = str(e).lower()
        if any(term in error_msg for term in ["timeout", "connection", "network"]):
            raise Retry(defer=ctx.get("job_try", 1) * 10) from e
        raise

    finally:
        await valkey.close()


async def detect_video_task(
    ctx: dict[str, Any],
    *,
    task_id: str,
    video_key: str,
    tracking: str = "auto",
) -> dict[str, Any]:
    """arq task: detect persons in uploaded video."""
    settings = get_settings()
    valkey = await get_valkey_client()

    try:
        now = datetime.now(UTC).isoformat()
        await valkey.hset(
            f"task:{task_id}",
            mapping={"status": TaskStatus.RUNNING, "started_at": now},
        )

        import tempfile
        from pathlib import Path

        import cv2

        from backend.app.storage import download_file
        from skating_ml.device import DeviceConfig
        from skating_ml.pose_estimation.rtmlib_extractor import RTMPoseExtractor
        from skating_ml.utils.video import get_video_meta
        from skating_ml.web_helpers import render_person_preview

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "input.mp4"
            download_file(video_key, str(video_path))

            cfg = DeviceConfig.default()
            extractor = RTMPoseExtractor(
                mode="balanced",
                tracking_backend="rtmlib",
                tracking_mode=tracking,
                conf_threshold=0.3,
                output_format="normalized",
                device=cfg.device,
            )
            persons, _ = extractor.preview_persons(video_path, num_frames=30)

            if not persons:
                result_data = {
                    "persons": [],
                    "preview_image": "",
                    "video_key": video_key,
                    "auto_click": None,
                    "status": "Люди не найдены. Попробуйте другое видео.",
                }
                await store_result(task_id, result_data, valkey=valkey)
                return result_data

            cap = cv2.VideoCapture(str(video_path))
            ret, frame = cap.read()
            cap.release()

            if not ret:
                raise RuntimeError("Failed to read video frame")

            meta = get_video_meta(video_path)
            w, h = meta.width, meta.height

            annotated = render_person_preview(frame, persons, selected_idx=None)
            success, buf = cv2.imencode(".png", annotated)
            if not success:
                raise RuntimeError("Failed to encode preview image")
            import base64

            preview_b64 = base64.b64encode(buf).decode("ascii")

            auto_click = None
            status_msg: str
            if len(persons) == 1:
                mid_hip = persons[0]["mid_hip"]
                auto_click = {"x": int(mid_hip[0] * w), "y": int(mid_hip[1] * h)}
                status_msg = "Обнаружен 1 человек — выбран автоматически"
            else:
                status_msg = f"Обнаружено {len(persons)} человек. Выберите на превью или из списка."

            persons_out = [
                {
                    "track_id": p["track_id"],
                    "hits": p["hits"],
                    "bbox": p["bbox"],
                    "mid_hip": p["mid_hip"],
                }
                for p in persons
            ]

            result_data = {
                "persons": persons_out,
                "preview_image": preview_b64,
                "video_key": video_key,
                "auto_click": auto_click,
                "status": status_msg,
            }
            await store_result(task_id, result_data, valkey=valkey)
            return result_data

    except Exception as e:
        logger.exception("Detection task %s failed", task_id)
        await store_error(task_id, str(e), valkey=valkey)
        raise
    finally:
        await valkey.close()


_settings = get_settings()


class WorkerSettings:
    """arq worker configuration."""

    queue_name: str = "skating:queue"
    max_jobs: int = _settings.app.worker_max_jobs
    retry_jobs: bool = True
    retry_delays: ClassVar[list[int]] = _settings.app.worker_retry_delays

    on_startup = startup
    on_shutdown = shutdown

    functions: ClassVar[list] = [process_video_task, detect_video_task]
    cron_jobs: ClassVar[list] = []

    redis_settings = RedisSettings(
        host=_settings.valkey.host,
        port=_settings.valkey.port,
        database=_settings.valkey.db,
        password=_settings.valkey.password.get_secret_value(),
    )
