"""Tests for R2 storage operations."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import ANY, MagicMock, patch

import pytest

# Mock aiobotocore before importing app.storage (it's a top-level import)
_mock_aiobotocore = ModuleType("aiobotocore")
_mock_aiobotocore.session = ModuleType("aiobotocore.session")
sys.modules["aiobotocore"] = _mock_aiobotocore
sys.modules["aiobotocore.session"] = _mock_aiobotocore.session

from app.storage import (  # noqa: E402
    delete_object,
    download_file,
    get_object_url,
    list_objects,
    object_exists,
    stream_object,
    upload_bytes,
    upload_file,
)

# --- Sync storage tests ---


@patch("app.storage._client")
def test_upload_file_calls_s3(mock_client):
    mock_s3 = MagicMock()
    mock_client.return_value = mock_s3
    with patch("app.storage.get_settings"):
        upload_file("/tmp/test.mp4", "input/test.mp4")
    mock_s3.upload_file.assert_called_once()


@patch("app.storage._client")
def test_download_file_calls_s3(mock_client):
    mock_s3 = MagicMock()
    mock_client.return_value = mock_s3
    with patch("app.storage.get_settings"):
        download_file("output/test.mp4", "/tmp/result.mp4")
    mock_s3.download_file.assert_called_once()


@patch("app.storage._client")
def test_delete_object_calls_s3(mock_client):
    mock_s3 = MagicMock()
    mock_client.return_value = mock_s3
    with patch("app.storage.get_settings"):
        delete_object("input/test.mp4")
    mock_s3.delete_object.assert_called_once()


@patch("app.storage._client")
def test_upload_bytes_calls_s3(mock_client):
    mock_s3 = MagicMock()
    mock_client.return_value = mock_s3
    with patch("app.storage.get_settings"):
        result = upload_bytes(b"hello world", "input/test.txt")
    assert result == "input/test.txt"
    mock_s3.put_object.assert_called_once_with(
        Bucket=ANY, Key="input/test.txt", Body=b"hello world"
    )


@patch("app.storage._client")
def test_stream_object_calls_s3(mock_client):
    mock_s3 = MagicMock()
    mock_body = MagicMock()
    mock_body.iter_chunks.return_value = [b"chunk1", b"chunk2"]
    mock_body.content_length = 12
    mock_body.content_type = "video/mp4"
    mock_s3.get_object.return_value = {
        "Body": mock_body,
        "ContentLength": 12,
        "ContentType": "video/mp4",
    }
    mock_client.return_value = mock_s3
    with patch("app.storage.get_settings"):
        body, length, ctype = stream_object("output/test.mp4")
    mock_s3.get_object.assert_called_once_with(Bucket=ANY, Key="output/test.mp4")
    assert length == 12
    assert ctype == "video/mp4"
    assert list(body.iter_chunks()) == [b"chunk1", b"chunk2"]


@patch("app.storage._client")
def test_object_exists_true(mock_client):
    mock_s3 = MagicMock()
    mock_s3.head_object.return_value = {"ContentLength": 100}
    mock_client.return_value = mock_s3
    with patch("app.storage.get_settings"):
        assert object_exists("output/test.mp4") is True
    mock_s3.head_object.assert_called_once()


@patch("app.storage._client")
def test_object_exists_false(mock_client):
    from botocore.exceptions import ClientError

    mock_s3 = MagicMock()
    mock_s3.head_object.side_effect = ClientError({"Error": {"Code": "404"}}, "head_object")
    mock_client.return_value = mock_s3
    with patch("app.storage.get_settings"):
        assert object_exists("output/missing.mp4") is False


@patch("app.storage._client")
def test_get_object_url_calls_s3(mock_client):
    mock_s3 = MagicMock()
    mock_s3.generate_presigned_url.return_value = "https://r2.example.com/output/test.mp4?sig=abc"
    mock_client.return_value = mock_s3
    with patch("app.storage.get_settings"):
        url = get_object_url("output/test.mp4", expires=1800)
    assert "sig=abc" in url
    mock_s3.generate_presigned_url.assert_called_once()


@patch("app.storage._client")
def test_list_objects(mock_client):
    mock_s3 = MagicMock()
    mock_s3.list_objects_v2.return_value = {
        "Contents": [{"Key": "input/a.mp4"}, {"Key": "input/b.mp4"}]
    }
    mock_client.return_value = mock_s3
    with patch("app.storage.get_settings"):
        keys = list_objects("input/")
    assert keys == ["input/a.mp4", "input/b.mp4"]


# --- Presigned URL tests ---


class TestPresignedURL:
    """Tests for presigned URL generation."""

    def test_get_object_url_default_method(self):
        """Should generate GET presigned URL by default."""
        with patch("app.storage._client") as mock_client:
            mock_client.return_value.generate_presigned_url.return_value = (
                "https://test.r2.dev/test-bucket/test-key?signature=abc"
            )

            url = get_object_url("test-key")

            mock_client.return_value.generate_presigned_url.assert_called_once()
            call_args = mock_client.return_value.generate_presigned_url.call_args
            assert call_args[0][0] == "get_object"
            assert call_args[1]["Params"]["Key"] == "test-key"

    def test_get_object_url_put_method(self):
        """Should generate PUT presigned URL when requested."""
        pytest.skip("PUT method not yet implemented in get_object_url")

    def test_get_object_url_custom_expires(self):
        """Should respect custom expiration time."""
        with patch("app.storage._client") as mock_client:
            mock_client.return_value.generate_presigned_url.return_value = "url"

            url = get_object_url("test-key", expires=7200)

            call_args = mock_client.return_value.generate_presigned_url.call_args
            assert call_args[1]["ExpiresIn"] == 7200


# --- Async tests (stubs) ---


class TestAsyncUpload:
    """Tests for async upload operations."""

    @pytest.mark.asyncio
    async def test_upload_file_async(self):
        pytest.skip("upload_file_async not yet implemented")

    @pytest.mark.asyncio
    async def test_upload_bytes_async(self):
        pytest.skip("upload_bytes_async not yet implemented")


class TestAsyncDownload:
    """Tests for async download operations."""

    @pytest.mark.asyncio
    async def test_download_file_async(self):
        pytest.skip("download_file_async not yet implemented")
