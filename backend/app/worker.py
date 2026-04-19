"""arq worker for video processing pipeline.

Run with: uv run python -m app.worker

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

from app.config import get_settings
from app.storage import download_file
from app.task_manager import (
    TaskStatus,
    get_valkey_client,
    is_cancelled,
    mark_cancelled,
    publish_task_event,
    store_error,
    store_result,
    update_progress,
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
    # Extract keypoint arrays (vectorized)
    # H36Key indices: RHIP=1, RKNEE=2, RFOOT=3, LHIP=4, LKNEE=5, LFOOT=6
    # SPINE=7, THORAX=8, NECK=9, HIP_CENTER=0
    r_hip = poses[:, H36Key.RHIP]  # (N, 3)
    r_knee = poses[:, H36Key.RKNEE]
    r_foot = poses[:, H36Key.RFOOT]
    l_hip = poses[:, H36Key.LHIP]
    l_knee = poses[:, H36Key.LKNEE]
    l_foot = poses[:, H36Key.LFOOT]
    thorax = poses[:, H36Key.THORAX]
    spine = poses[:, H36Key.SPINE]
    neck = poses[:, H36Key.NECK]
    hip_center = poses[:, H36Key.HIP_CENTER]

    # Helper function to compute angles between vectors
    def compute_angles_batch(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Compute angles at point b for vectors (a->b) and (b->c).

        Args:
            a, b, c: (N, 3) arrays of keypoints

        Returns:
            (N,) array of angles in degrees, with NaN for invalid frames
        """
        vec1 = b - a  # (N, 3)
        vec2 = c - b  # (N, 3)

        # Compute norms
        norm1 = np.linalg.norm(vec1, axis=1)
        norm2 = np.linalg.norm(vec2, axis=1)

        # Dot product
        dot = np.sum(vec1 * vec2, axis=1)

        # Cosine with clipping
        cos = np.clip(dot / (norm1 * norm2 + 1e-8), -1, 1)

        # Convert to degrees
        angles = np.degrees(np.arccos(cos))

        # Mark invalid frames (where any keypoint is NaN)
        valid_mask = ~(np.isnan(a).any(axis=1) | np.isnan(b).any(axis=1) | np.isnan(c).any(axis=1))
        angles[~valid_mask] = np.nan

        return angles

    # Knee angles (hip-knee-ankle)
    knee_angles_r = compute_angles_batch(r_hip, r_knee, r_foot)
    knee_angles_l = compute_angles_batch(l_hip, l_knee, l_foot)

    # Hip angles (thorax-hip-knee)
    hip_angles_r = compute_angles_batch(thorax, r_hip, r_knee)
    hip_angles_l = compute_angles_batch(thorax, l_hip, l_knee)

    # Trunk lean (spine angle from vertical)
    spine_vec = neck - spine  # (N, 3)
    spine_vec[:, 1] = 0  # Project to horizontal plane (set y to 0)

    # Compute lean angle: arctan2(x, z)
    trunk_lean = np.degrees(np.arctan2(spine_vec[:, 0], spine_vec[:, 2]))

    # Handle division by zero (when z=0)
    z_zero = spine_vec[:, 2] == 0
    trunk_lean[z_zero] = 0.0

    # Mark invalid frames
    valid_spine = ~(np.isnan(spine).any(axis=1) | np.isnan(neck).any(axis=1))
    trunk_lean[~valid_spine] = np.nan

    # CoM height (hip center y-coordinate)
    com_height = hip_center[:, 1].copy()
    valid_hip = ~np.isnan(hip_center[:, 1])
    com_height[~valid_hip] = np.nan

    # Convert to lists for JSON (NaN -> None)
    def to_list(arr: np.ndarray) -> list:
        """Convert numpy array to list, replacing NaN with None."""
        return [float(x) if not np.isnan(x) else None for x in arr]

    return {
        "knee_angles_r": to_list(knee_angles_r),
        "knee_angles_l": to_list(knee_angles_l),
        "hip_angles_r": to_list(hip_angles_r),
        "hip_angles_l": to_list(hip_angles_l),
        "trunk_lean": to_list(trunk_lean),
        "com_height": to_list(com_height),
    }


async def startup(ctx: dict[str, Any]) -> None:
    logger.info("Video processing worker starting up")


async def shutdown(ctx: dict[str, Any]) -> None:
    logger.info("Video processing worker shutting down")
    pool = ctx.get("redis")
    if pool:
        await pool.close()
        logger.info("Worker valkey pool closed")


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
        await update_progress(task_id, 0.0, "Starting...", valkey=valkey)
        await publish_task_event(
            task_id, {"status": "running", "progress": 0.0, "message": "Starting..."}, valkey=valkey
        )

        from app.crud.session import get_by_id
        from app.database import async_session  # type: ignore[import-untyped]
        from app.vastai.client import process_video_remote_async

        # Fetch element_type from session if session_id provided
        element_type = None
        if session_id:
            async with async_session() as db:
                session = await get_by_id(db, session_id)
                if session:
                    element_type = session.element_type

        logger.info("Dispatching task %s to Vast.ai (video_key=%s)", task_id, video_key)
        await update_progress(task_id, 0.1, "Dispatching to GPU...", valkey=valkey)
        await publish_task_event(
            task_id,
            {"status": "running", "progress": 0.1, "message": "Dispatching to GPU..."},
            valkey=valkey,
        )

        # Cancellation check before expensive GPU dispatch
        if await is_cancelled(task_id, valkey=valkey):
            await mark_cancelled(task_id, valkey=valkey)
            return {"status": "cancelled"}

        vast_result = await process_video_remote_async(
            video_key=video_key,
            person_click={"x": person_click["x"], "y": person_click["y"]} if person_click else None,
            frame_skip=frame_skip,
            layer=layer,
            tracking=tracking,
            export=export,
            ml_flags=ml_flags,
            element_type=element_type,
        )
        logger.info("Vast.ai processing complete for task %s", task_id)
        await update_progress(task_id, 0.7, "GPU processing complete", valkey=valkey)
        await publish_task_event(
            task_id,
            {"status": "running", "progress": 0.7, "message": "GPU processing complete"},
            valkey=valkey,
        )

        # Cancellation check after GPU returns (skip post-processing)
        if await is_cancelled(task_id, valkey=valkey):
            await mark_cancelled(task_id, valkey=valkey)
            return {"status": "cancelled"}

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
                    await update_progress(task_id, 0.85, "Preparing results...", valkey=valkey)
                    await publish_task_event(
                        task_id,
                        {"status": "running", "progress": 0.85, "message": "Preparing results..."},
                        valkey=valkey,
                    )
            except Exception as pose_err:  # noqa: BLE001
                logger.warning("Failed to prepare pose data: %s", pose_err)

        response_data = {
            "video_path": vast_result.video_key,
            "stats": vast_result.stats,
            "status": "Analysis complete!",
        }
        await store_result(task_id, response_data, valkey=valkey)
        await update_progress(task_id, 1.0, "Done", valkey=valkey)
        await publish_task_event(
            task_id, {"status": "completed", "progress": 1.0, "message": "Done"}, valkey=valkey
        )
        if session_id and vast_result.metrics:
            try:
                from app.crud.session import update_session_analysis
                from app.database import async_session  # type: ignore[import-untyped]
                from app.services.session_saver import save_analysis_results

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
        try:
            await publish_task_event(
                task_id, {"status": "failed", "progress": 0.0, "message": str(e)}, valkey=valkey
            )
        except Exception:  # noqa: BLE001
            logger.warning("Failed to publish error event for task %s", task_id)
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
        await update_progress(task_id, 0.0, "Starting detection...", valkey=valkey)
        await publish_task_event(
            task_id,
            {"status": "running", "progress": 0.0, "message": "Starting detection..."},
            valkey=valkey,
        )

        import tempfile
        from pathlib import Path

        import cv2

        from app.storage import download_file
        from src.device import DeviceConfig
        from src.pose_estimation.pose_extractor import PoseExtractor
        from src.utils.video import get_video_meta
        from src.web_helpers import render_person_preview

        await update_progress(task_id, 0.1, "Downloading video...", valkey=valkey)
        await publish_task_event(
            task_id,
            {"status": "running", "progress": 0.1, "message": "Downloading video..."},
            valkey=valkey,
        )

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
            await update_progress(task_id, 0.8, "Extracting poses...", valkey=valkey)
            await publish_task_event(
                task_id,
                {"status": "running", "progress": 0.8, "message": "Extracting poses..."},
                valkey=valkey,
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
            await update_progress(task_id, 1.0, "Done", valkey=valkey)
            await publish_task_event(
                task_id, {"status": "completed", "progress": 1.0, "message": "Done"}, valkey=valkey
            )
            return result_data

    except Exception as e:
        logger.exception("Detection task %s failed", task_id)
        await store_error(task_id, str(e), valkey=valkey)
        try:
            await publish_task_event(
                task_id, {"status": "failed", "progress": 0.0, "message": str(e)}, valkey=valkey
            )
        except Exception:  # noqa: BLE001
            logger.warning("Failed to publish error event for task %s", task_id)
        raise
    finally:
        await valkey.close()


async def analyze_music_task(
    ctx: dict[str, Any],
    *,
    music_id: str,
    r2_key: str,
) -> dict[str, Any]:
    """arq task: analyze music file for BPM, structure, and energy peaks.

    Args:
        music_id: Database ID of the music record
        r2_key: R2 storage key for the audio file

    Returns:
        dict with status and analysis results
    """
    valkey = await get_valkey_client()

    try:
        from app.crud.choreography import (
            find_music_by_fingerprint,
            get_music_analysis_by_id,
            update_music_analysis,
        )
        from app.database import async_session_factory
        from app.services.choreography.fingerprint import compute_fingerprint
        from app.services.choreography.music_analyzer import analyze_music_sync

        logger.info("Starting music analysis for music_id=%s, r2_key=%s", music_id, r2_key)

        # Download from R2 to temp file
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / f"music_{music_id}.mp3"
            logger.info("Downloading music from R2: %s -> %s", r2_key, audio_path)
            await asyncio.to_thread(download_file, r2_key, str(audio_path))

            # Compute fingerprint
            logger.info("Computing fingerprint for %s", audio_path)
            fingerprint = await asyncio.to_thread(compute_fingerprint, str(audio_path))
            if not fingerprint:
                raise RuntimeError("Failed to compute fingerprint")

            logger.info("Fingerprint computed: %s", fingerprint[:16] + "...")

            # Check for duplicate
            async with async_session_factory() as db:
                music = await get_music_analysis_by_id(db, music_id)
                if not music:
                    raise RuntimeError(f"Music record {music_id} not found")

                # Store fingerprint
                await update_music_analysis(db, music, fingerprint=fingerprint)
                await db.commit()

                # Check for existing analysis with same fingerprint
                duplicate = await find_music_by_fingerprint(db, fingerprint)
                if duplicate and duplicate.id != music_id:
                    fp_preview = (
                        duplicate.fingerprint[:16] + "..." if duplicate.fingerprint else "N/A"
                    )
                    logger.info(
                        "Found duplicate analysis: %s (music_id=%s)", duplicate.id, fp_preview
                    )
                    # Copy analysis results from duplicate
                    await update_music_analysis(
                        db,
                        music,
                        audio_url=music.audio_url,  # Keep our own URL
                        duration_sec=duplicate.duration_sec,
                        bpm=duplicate.bpm,
                        peaks=duplicate.peaks,
                        structure=duplicate.structure,
                        energy_curve=duplicate.energy_curve,
                        status="completed",
                    )
                    await db.commit()
                    return {
                        "status": "completed",
                        "music_id": music_id,
                        "duplicate_of": duplicate.id,
                        "bpm": duplicate.bpm,
                        "duration_sec": duplicate.duration_sec,
                    }

                # No duplicate - run full analysis
                logger.info("No duplicate found, running full music analysis")
                result = await asyncio.to_thread(analyze_music_sync, str(audio_path))

                await update_music_analysis(
                    db,
                    music,
                    audio_url=f"/files/{r2_key}",
                    duration_sec=result["duration_sec"],
                    bpm=result["bpm"],
                    peaks=result["peaks"],
                    structure=result.get("structure") or [],
                    energy_curve=result["energy_curve"],
                    status="completed",
                )
                await db.commit()

                logger.info(
                    "Music analysis complete: music_id=%s, bpm=%.1f, duration=%.1f",
                    music_id,
                    result["bpm"],
                    result["duration_sec"],
                )

                return {
                    "status": "completed",
                    "music_id": music_id,
                    "bpm": result["bpm"],
                    "duration_sec": result["duration_sec"],
                }

    except Exception as e:
        logger.exception("Music analysis task failed for music_id=%s", music_id)

        # Update DB status to failed
        try:
            from app.crud.choreography import get_music_analysis_by_id, update_music_analysis
            from app.database import async_session_factory

            async with async_session_factory() as db:
                music = await get_music_analysis_by_id(db, music_id)
                if music:
                    await update_music_analysis(db, music, status="failed")
                    await db.commit()
        except Exception:  # noqa: BLE001
            logger.warning("Failed to update music status to failed")

        raise

    finally:
        await valkey.close()


_settings = get_settings()


class FastWorkerSettings:
    """arq worker for lightweight detection tasks."""

    queue_name: str = "skating:queue:fast"
    max_jobs: int = (
        _settings.app.worker_max_jobs_remote
        if _settings.vastai.api_key.get_secret_value()
        else _settings.app.worker_max_jobs
    )
    retry_jobs: bool = True
    retry_delays: ClassVar[list[int]] = _settings.app.worker_retry_delays
    job_completion_wait: int = 120

    on_startup = startup
    on_shutdown = shutdown
    functions: ClassVar[list] = [detect_video_task, analyze_music_task]
    cron_jobs: ClassVar[list] = []

    redis_settings = RedisSettings(
        host=_settings.valkey.host,
        port=_settings.valkey.port,
        database=_settings.valkey.db,
        password=_settings.valkey.password.get_secret_value(),
    )


class HeavyWorkerSettings:
    """arq worker for full ML pipeline processing."""

    queue_name: str = "skating:queue:heavy"
    max_jobs: int = 1  # GPU-bound, can't parallelize
    retry_jobs: bool = True
    retry_delays: ClassVar[list[int]] = _settings.app.worker_retry_delays
    job_completion_wait: int = 600

    on_startup = startup
    on_shutdown = shutdown
    functions: ClassVar[list] = [process_video_task]
    cron_jobs: ClassVar[list] = []

    redis_settings = RedisSettings(
        host=_settings.valkey.host,
        port=_settings.valkey.port,
        database=_settings.valkey.db,
        password=_settings.valkey.password.get_secret_value(),
    )
