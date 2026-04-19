"""S3-compatible object storage (Cloudflare R2) for video transfer."""

from __future__ import annotations

import logging
from pathlib import Path

import aiobotocore.session
import boto3
from botocore.config import Config as BotoConfig

from app.config import get_settings

logger = logging.getLogger(__name__)

# Async session factory
_async_session = aiobotocore.session.get_session()


def _client():
    s = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=s.r2.endpoint_url or None,
        aws_access_key_id=s.r2.access_key_id.get_secret_value(),
        aws_secret_access_key=s.r2.secret_access_key.get_secret_value(),
        config=BotoConfig(signature_version="s3v4"),
        region_name="auto",
    )


def upload_file(local_path: str | Path, key: str) -> str:
    """Upload file to R2. Returns the key."""
    bucket = get_settings().r2.bucket
    logger.info("Uploading %s -> s3://%s/%s", local_path, bucket, key)
    _client().upload_file(str(local_path), bucket, key)
    return key


def download_file(key: str, local_path: str | Path) -> str:
    """Download file from R2. Returns the local path."""
    bucket = get_settings().r2.bucket
    logger.info("Downloading s3://%s/%s -> %s", bucket, key, local_path)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    _client().download_file(bucket, key, str(local_path))
    return str(local_path)


def delete_object(key: str) -> None:
    """Delete object from R2."""
    _client().delete_object(Bucket=get_settings().r2.bucket, Key=key)


def upload_bytes(data: bytes, key: str) -> str:
    """Upload bytes to R2. Returns the key."""
    bucket = get_settings().r2.bucket
    logger.info("Uploading %d bytes -> s3://%s/%s", len(data), bucket, key)
    _client().put_object(Bucket=bucket, Key=key, Body=data)
    return key


def stream_object(key: str) -> tuple:
    """Stream object from R2. Returns (body, content_length, content_type)."""
    bucket = get_settings().r2.bucket
    logger.info("Streaming s3://%s/%s", bucket, key)
    resp = _client().get_object(Bucket=bucket, Key=key)
    body = resp["Body"]
    length = resp.get("ContentLength", 0)
    ctype = resp.get("ContentType", "application/octet-stream")
    return body, length, ctype


def object_exists(key: str) -> bool:
    """Check if object exists in R2."""
    from botocore.exceptions import ClientError

    try:
        _client().head_object(Bucket=get_settings().r2.bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise


def get_object_url(key: str, expires: int = 3600) -> str:
    """Generate a presigned URL for an object."""
    bucket = get_settings().r2.bucket
    return _client().generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires,
    )


def list_objects(prefix: str) -> list[str]:
    """List object keys with given prefix."""
    bucket = get_settings().r2.bucket
    resp = _client().list_objects_v2(Bucket=bucket, Prefix=prefix)
    return [obj["Key"] for obj in resp.get("Contents", [])]


# ============ Async versions for GPU server ============


async def _async_client():
    """Async S3 client factory."""
    s = get_settings()
    return _async_session.create_client(
        "s3",
        endpoint_url=s.r2.endpoint_url or None,
        aws_access_key_id=s.r2.access_key_id.get_secret_value(),
        aws_secret_access_key=s.r2.secret_access_key.get_secret_value(),
        config=BotoConfig(signature_version="s3v4"),
        region_name="auto",
    )


async def upload_file_async(local_path: str | Path, key: str) -> str:
    """Upload file to R2 asynchronously. Returns the key."""
    bucket = get_settings().r2.bucket
    logger.info("Uploading %s -> s3://%s/%s (async)", local_path, bucket, key)
    async with await _async_client() as s3:
        await s3.upload_file(str(local_path), bucket, key)
    return key


async def download_file_async(key: str, local_path: str | Path) -> str:
    """Download file from R2 asynchronously. Returns the local path."""
    bucket = get_settings().r2.bucket
    logger.info("Downloading s3://%s/%s -> %s (async)", bucket, key, local_path)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    async with await _async_client() as s3:
        await s3.download_file(bucket, key, str(local_path))
    return str(local_path)


async def upload_bytes_async(data: bytes, key: str) -> str:
    """Upload bytes to R2 asynchronously. Returns the key."""
    bucket = get_settings().r2.bucket
    logger.info("Uploading %d bytes -> s3://%s/%s (async)", len(data), bucket, key)
    async with await _async_client() as s3:
        await s3.put_object(Bucket=bucket, Key=key, Body=data)
    return key


async def object_exists_async(key: str) -> bool:
    """Check if object exists in R2 asynchronously."""
    from botocore.exceptions import ClientError

    async with await _async_client() as s3:
        try:
            await s3.head_object(Bucket=get_settings().r2.bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise


async def stream_object_async(key: str) -> tuple:
    """Stream object from R2 asynchronously. Returns (body, content_length, content_type)."""
    bucket = get_settings().r2.bucket
    logger.info("Streaming s3://%s/%s (async)", bucket, key)
    async with await _async_client() as s3:
        resp = await s3.get_object(Bucket=bucket, Key=key)
        body = resp["Body"]
        length = resp.get("ContentLength", 0)
        ctype = resp.get("ContentType", "application/octet-stream")
        return body, length, ctype


async def get_object_url_async(key: str, expires: int = 3600) -> str:
    """Generate a presigned URL for an object asynchronously."""
    bucket = get_settings().r2.bucket
    async with await _async_client() as s3:
        return await s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires,
        )


async def delete_object_async(key: str) -> None:
    """Delete object from R2 asynchronously."""
    async with await _async_client() as s3:
        await s3.delete_object(Bucket=get_settings().r2.bucket, Key=key)


async def list_objects_async(prefix: str) -> list[str]:
    """List object keys with given prefix asynchronously."""
    async with await _async_client() as s3:
        bucket = get_settings().r2.bucket
        resp = await s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        return [obj["Key"] for obj in resp.get("Contents", [])]
