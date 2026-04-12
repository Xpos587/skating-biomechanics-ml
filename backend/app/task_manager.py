"""Valkey-backed task state management for video processing jobs."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import redis.asyncio as aioredis

from backend.app.config import get_settings

logger = logging.getLogger(__name__)

TASK_KEY_PREFIX = "task:"
TASK_CANCEL_PREFIX = "task_cancel:"


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


async def get_valkey_client() -> aioredis.Redis:
    settings = get_settings()
    return aioredis.Redis(
        host=settings.valkey.host,
        port=settings.valkey.port,
        db=settings.valkey.db,
        password=settings.valkey.password.get_secret_value(),
        decode_responses=True,
    )


async def create_task_state(
    task_id: str,
    video_key: str,
    valkey: aioredis.Redis | None = None,
) -> None:
    close = valkey is None
    if valkey is None:
        valkey = await get_valkey_client()
    try:
        ttl = get_settings().app.task_ttl_seconds
        now = datetime.now(UTC).isoformat()
        await valkey.hset(
            f"{TASK_KEY_PREFIX}{task_id}",
            mapping={
                "task_id": task_id,
                "status": TaskStatus.PENDING,
                "video_key": video_key,
                "progress": "0.0",
                "message": "Queued",
                "created_at": now,
                "started_at": "",
                "completed_at": "",
                "error": "",
            },
        )
        await valkey.expire(f"{TASK_KEY_PREFIX}{task_id}", ttl)
    finally:
        if close:
            await valkey.close()


async def update_progress(
    task_id: str,
    fraction: float,
    message: str,
    valkey: aioredis.Redis | None = None,
) -> None:
    close = valkey is None
    if valkey is None:
        valkey = await get_valkey_client()
    try:
        await valkey.hset(
            f"{TASK_KEY_PREFIX}{task_id}",
            mapping={"progress": str(round(fraction, 3)), "message": message},
        )
    finally:
        if close:
            await valkey.close()


async def store_result(
    task_id: str,
    result: dict[str, Any],
    valkey: aioredis.Redis | None = None,
) -> None:
    close = valkey is None
    if valkey is None:
        valkey = await get_valkey_client()
    try:
        now = datetime.now(UTC).isoformat()
        await valkey.hset(
            f"{TASK_KEY_PREFIX}{task_id}",
            mapping={
                "status": TaskStatus.COMPLETED,
                "progress": "1.0",
                "message": "Done",
                "completed_at": now,
                "result": json.dumps(result),
            },
        )
    finally:
        if close:
            await valkey.close()


async def store_error(
    task_id: str,
    error_message: str,
    valkey: aioredis.Redis | None = None,
) -> None:
    close = valkey is None
    if valkey is None:
        valkey = await get_valkey_client()
    try:
        now = datetime.now(UTC).isoformat()
        await valkey.hset(
            f"{TASK_KEY_PREFIX}{task_id}",
            mapping={
                "status": TaskStatus.FAILED,
                "completed_at": now,
                "error": error_message,
            },
        )
    finally:
        if close:
            await valkey.close()


async def mark_cancelled(task_id: str, valkey: aioredis.Redis | None = None) -> None:
    close = valkey is None
    if valkey is None:
        valkey = await get_valkey_client()
    try:
        now = datetime.now(UTC).isoformat()
        await valkey.hset(
            f"{TASK_KEY_PREFIX}{task_id}",
            mapping={
                "status": TaskStatus.CANCELLED,
                "completed_at": now,
                "message": "Cancelled",
            },
        )
    finally:
        if close:
            await valkey.close()


async def get_task_state(
    task_id: str, valkey: aioredis.Redis | None = None
) -> dict[str, Any] | None:
    close = valkey is None
    if valkey is None:
        valkey = await get_valkey_client()
    try:
        data = await valkey.hgetall(f"{TASK_KEY_PREFIX}{task_id}")
        if not data:
            return None
        result = data.get("result")
        data["result"] = json.loads(result) if result else None
        data["progress"] = float(data.get("progress", "0"))
        return data
    finally:
        if close:
            await valkey.close()


async def is_cancelled(task_id: str, valkey: aioredis.Redis | None = None) -> bool:
    close = valkey is None
    if valkey is None:
        valkey = await get_valkey_client()
    try:
        return await valkey.get(f"{TASK_CANCEL_PREFIX}{task_id}") == "1"
    finally:
        if close:
            await valkey.close()


async def set_cancel_signal(task_id: str, valkey: aioredis.Redis | None = None) -> None:
    close = valkey is None
    if valkey is None:
        valkey = await get_valkey_client()
    try:
        ttl = get_settings().app.task_ttl_seconds
        await valkey.setex(f"{TASK_CANCEL_PREFIX}{task_id}", ttl, "1")
    finally:
        if close:
            await valkey.close()
