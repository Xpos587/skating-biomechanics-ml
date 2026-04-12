"""Chunked S3 multipart upload endpoints."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from backend.app.config import get_settings
from backend.app.storage import _client

if TYPE_CHECKING:
    from backend.app.auth.deps import CurrentUser

router = APIRouter(tags=["uploads"])

CHUNK_SIZE = 5 * 1024 * 1024  # 5MB


@router.post("/uploads/init")
async def init_upload(
    user: CurrentUser,
    file_name: str = Query(..., min_length=1),
    content_type: str = Query("video/mp4"),
    total_size: int = Query(..., gt=0),
):
    """Initialize a multipart upload. Returns upload_id and pre-signed part URLs."""
    r2 = _client()
    bucket = get_settings().r2.bucket
    key = f"uploads/{user.id}/{uuid.uuid4()}/{file_name}"

    upload_id = r2.create_multipart_upload(
        Bucket=bucket,
        Key=key,
        ContentType=content_type,
    )["UploadId"]

    # Calculate number of parts
    part_count = (total_size + CHUNK_SIZE - 1) // CHUNK_SIZE

    # Generate pre-signed URLs for each part
    part_urls = []
    for part_number in range(1, part_count + 1):
        url = r2.generate_presigned_url(
            ClientMethod="upload_part",
            Params={
                "Bucket": bucket,
                "Key": key,
                "UploadId": upload_id,
                "PartNumber": part_number,
            },
            ExpiresIn=3600,
        )
        part_urls.append({"part_number": part_number, "url": url})

    return {
        "upload_id": upload_id,
        "key": key,
        "chunk_size": CHUNK_SIZE,
        "part_count": part_count,
        "parts": part_urls,
    }


class CompleteUploadRequest(BaseModel):
    upload_id: str
    key: str
    parts: list[dict]


@router.post("/uploads/complete")
async def complete_upload(user: CurrentUser, body: CompleteUploadRequest):
    """Complete a multipart upload. Returns the final object key."""
    r2 = _client()
    bucket = get_settings().r2.bucket

    multipart_parts = [
        {"PartNumber": p["part_number"], "ETag": p["etag"]}
        for p in sorted(body.parts, key=lambda x: x["part_number"])
    ]

    if not multipart_parts:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No parts provided")

    r2.complete_multipart_upload(
        Bucket=bucket,
        Key=body.key,
        UploadId=body.upload_id,
        MultipartUpload={"Parts": multipart_parts},
    )

    return {"status": "completed", "key": body.key}
