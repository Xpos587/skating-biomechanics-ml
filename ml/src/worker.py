"""arq worker for video processing pipeline.

Run with: uv run python -m src.worker

Dispatches all processing to Vast.ai Serverless GPU.
No local GPU fallback.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from arq import Retry
from arq.connections import RedisSettings

from backend.app.config import get_settings
from backend.app.storage import download_file
from backend.app.task_manager import (
    TaskStatus,
    get_valkey_client,
    store_error,
    store_result,
)
from src.types import H36Key

logger = logging.getLogger(__name__)

# Configure OpenMP threads for better CPU performance
os.environ.setdefault("OMP_NUM_THREADS", "2")


def _sample_poses(
    poses: np.ndarray,
    sample_rate: int = 10,
) -> dict:
    """Sample poses to reduce data transfer for frontend.

    Args:
        poses: (N, 17, 3) array of poses
        sample_rate: Sample every Nth frame (default: 10)

    Returns:
        dict with frames list and poses array for sampled frames
    """
    n_frames = len(poses)
    sampled_indices = list(range(0, n_frames, sample_rate))

    # Extract sampled poses as list for JSON serialization
    sampled_poses = poses[sampled_indices].tolist()

    return {
        "frames": sampled_indices,
        "poses": sampled_poses,
    }


def _compute_frame_metrics(poses: np.ndarray) -> dict:
    """Compute frame-by-frame biomechanics metrics.

    Args:
        poses: (N, 17, 3) array of poses

    Returns:
        dict with metric arrays (knee angles, hip angles, trunk lean, CoM height)
    """
    knee_angles_r = []
    knee_angles_l = []
    hip_angles_r = []
    hip_angles_l = []
    trunk_lean = []
    com_height = []

    for pose in poses:
        # Knee angles (hip-knee-ankle)
        r_knee = pose[H36Key.RHIP]
        r_knee_joint = pose[H36Key.RKNEE]
        r_ankle = pose[H36Key.RFOOT]

        l_knee = pose[H36Key.LHIP]
        l_knee_joint = pose[H36Key.LKNEE]
        l_ankle = pose[H36Key.LFOOT]

        # Right knee angle
        if not (np.isnan(r_knee).any() or np.isnan(r_knee_joint).any() or np.isnan(r_ankle).any()):
            vec1 = r_knee_joint - r_knee
            vec2 = r_ankle - r_knee_joint
            angle = np.degrees(
                np.arccos(
                    np.clip(
                        np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8),
                        -1,
                        1,
                    )
                )
            )
            knee_angles_r.append(angle)
        else:
            knee_angles_r.append(float("nan"))

        # Left knee angle
        if not (np.isnan(l_knee).any() or np.isnan(l_knee_joint).any() or np.isnan(l_ankle).any()):
            vec1 = l_knee_joint - l_knee
            vec2 = l_ankle - l_knee_joint
            angle = np.degrees(
                np.arccos(
                    np.clip(
                        np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8),
                        -1,
                        1,
                    )
                )
            )
            knee_angles_l.append(angle)
        else:
            knee_angles_l.append(float("nan"))

        # Hip angles (thorax-hip-knee)
        r_thorax = pose[H36Key.THORAX]
        l_thorax = pose[H36Key.THORAX]

        # Right hip angle
        if not (np.isnan(r_thorax).any() or np.isnan(r_knee).any() or np.isnan(r_knee_joint).any()):
            vec1 = r_knee - r_thorax
            vec2 = r_knee_joint - r_knee
            angle = np.degrees(
                np.arccos(
                    np.clip(
                        np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8),
                        -1,
                        1,
                    )
                )
            )
            hip_angles_r.append(angle)
        else:
            hip_angles_r.append(float("nan"))

        # Left hip angle
        if not (np.isnan(l_thorax).any() or np.isnan(l_knee).any() or np.isnan(l_knee_joint).any()):
            vec1 = l_knee - l_thorax
            vec2 = l_knee_joint - l_knee
            angle = np.degrees(
                np.arccos(
                    np.clip(
                        np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8),
                        -1,
                        1,
                    )
                )
            )
            hip_angles_l.append(angle)
        else:
            hip_angles_l.append(float("nan"))

        # Trunk lean (spine angle from vertical)
        spine = pose[H36Key.SPINE]
        neck = pose[H36Key.NECK]
        if not (np.isnan(spine).any() or np.isnan(neck).any()):
            spine_vec = neck - spine
            spine_vec[1] = 0  # Project to horizontal plane
            lean = np.degrees(np.arctan2(spine_vec[0], spine_vec[2])) if spine_vec[2] != 0 else 0
            trunk_lean.append(lean)
        else:
            trunk_lean.append(float("nan"))

        # CoM height (hip center y)
        hip_center = pose[H36Key.HIP_CENTER]
        com_height.append(hip_center[1] if not np.isnan(hip_center[1]) else float("nan"))

    # Convert to lists for JSON
    return {
        "knee_angles_r": [float(x) if not np.isnan(x) else None for x in knee_angles_r],
        "knee_angles_l": [float(x) if not np.isnan(x) else None for x in knee_angles_l],
        "hip_angles_r": [float(x) if not np.isnan(x) else None for x in hip_angles_r],
        "hip_angles_l": [float(x) if not np.isnan(x) else None for x in hip_angles_l],
        "trunk_lean": [float(x) if not np.isnan(x) else None for x in trunk_lean],
        "com_height": [float(x) if not np.isnan(x) else None for x in com_height],
    }


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
        from backend.app.database import async_session  # type: ignore[import-untyped]
        from src.vastai.client import process_video_remote

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

        # Prepare pose data for JSON storage (if poses available)
        pose_data = None
        frame_metrics = None

        if vast_result.poses_key:
            try:
                import tempfile

                # Download poses temporarily for sampling
                with tempfile.TemporaryDirectory() as tmpdir:
                    poses_path = Path(tmpdir) / "poses.npy"
                    await asyncio.to_thread(download_file, vast_result.poses_key, str(poses_path))

                    # Load poses and prepare data
                    poses = np.load(str(poses_path))
                    fps = vast_result.stats.get("fps", 30.0)

                    # Run sampling and metrics computation in parallel
                    sample_future = asyncio.to_thread(_sample_poses, poses, 10)
                    metrics_future = asyncio.to_thread(_compute_frame_metrics, poses)
                    sampled, frame_metrics = await asyncio.gather(sample_future, metrics_future)
                    sampled["fps"] = fps
                    pose_data = sampled

                    logger.info(
                        "Prepared pose_data: %d frames, %d metrics",
                        len(sampled["frames"]),
                        len(poses),
                    )
            except Exception as pose_err:  # noqa: BLE001
                logger.warning("Failed to prepare pose data: %s", pose_err)

        response_data = {
            "video_path": vast_result.video_key,
            "stats": vast_result.stats,
            "status": "Analysis complete!",
        }
        await store_result(task_id, response_data, valkey=valkey)

        # Save analysis results to Postgres if session_id was provided
        if session_id and vast_result.metrics:
            try:
                from backend.app.crud.session import update_session_analysis
                from backend.app.database import async_session  # type: ignore[import-untyped]
                from backend.app.services.session_saver import save_analysis_results

                async with async_session() as db:
                    # Save pose data and frame metrics as JSON
                    if pose_data or frame_metrics:
                        await update_session_analysis(
                            db,
                            session_id=session_id,
                            pose_data=pose_data,
                            frame_metrics=frame_metrics,
                            phases=vast_result.phases,  # type: ignore[arg-type]
                        )

                    # Save metrics and recommendations (existing flow)
                    await save_analysis_results(
                        db,
                        session_id=session_id,
                        metrics=vast_result.metrics,
                        phases=vast_result.phases,
                        recommendations=vast_result.recommendations or [],
                    )
                    await db.commit()
            except Exception as save_err:  # noqa: BLE001
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
        from src.device import DeviceConfig
        from src.pose_estimation.pose_extractor import PoseExtractor
        from src.utils.video import get_video_meta
        from src.web_helpers import render_person_preview

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "input.mp4"
            await asyncio.to_thread(download_file, video_key, str(video_path))

            cfg = DeviceConfig.default()
            extractor = PoseExtractor(
                mode="balanced",
                tracking_backend="rtmlib",
                tracking_mode=tracking,
                conf_threshold=0.3,
                output_format="normalized",
                device=cfg.device,
            )
            persons, _ = await asyncio.to_thread(
                extractor.preview_persons, video_path, num_frames=30
            )

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

            annotated = render_person_preview(
                frame,  # type: ignore[arg-type]
                persons,  # type: ignore[arg-type]
                selected_idx=None,
            )
            success, buf = cv2.imencode(".png", annotated)
            if not success:
                raise RuntimeError("Failed to encode preview image")
            import base64

            preview_b64 = base64.b64encode(buf).decode("ascii")  # type: ignore[arg-type]

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
