"""Unit tests for S3 client retry and error handling."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from shared.s3_client import S3Client


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "test"}}, "TestOp")


@patch("shared.s3_client.boto3")
def test_upload_file_success(mock_boto3: MagicMock) -> None:
    mock_s3 = MagicMock()
    mock_boto3.client.return_value = mock_s3

    client = S3Client(bucket="test-bucket")
    client.upload_file("/tmp/file.bin", "models/file.bin")

    mock_s3.upload_file.assert_called_once_with("/tmp/file.bin", "test-bucket", "models/file.bin")


@patch("shared.s3_client.time.sleep")
@patch("shared.s3_client.boto3")
def test_upload_file_retries_on_transient_error(mock_boto3: MagicMock, mock_sleep: MagicMock) -> None:
    mock_s3 = MagicMock()
    mock_boto3.client.return_value = mock_s3
    mock_s3.upload_file.side_effect = [_client_error("InternalError"), None]

    client = S3Client(bucket="test-bucket")
    client.upload_file("/tmp/file.bin", "models/file.bin")

    assert mock_s3.upload_file.call_count == 2
    mock_sleep.assert_called_once_with(1.0)


@patch("shared.s3_client.time.sleep")
@patch("shared.s3_client.boto3")
def test_upload_file_fails_after_max_retries(mock_boto3: MagicMock, mock_sleep: MagicMock) -> None:
    mock_s3 = MagicMock()
    mock_boto3.client.return_value = mock_s3
    mock_s3.upload_file.side_effect = _client_error("ServiceUnavailable")

    client = S3Client(bucket="test-bucket")
    with pytest.raises(ClientError):
        client.upload_file("/tmp/file.bin", "models/file.bin")

    assert mock_s3.upload_file.call_count == 3


@patch("shared.s3_client.boto3")
def test_download_file_no_retry_on_not_found(mock_boto3: MagicMock) -> None:
    mock_s3 = MagicMock()
    mock_boto3.client.return_value = mock_s3
    mock_s3.download_file.side_effect = _client_error("NoSuchKey")

    client = S3Client(bucket="test-bucket")
    with pytest.raises(ClientError):
        client.download_file("missing/key", "/tmp/out")

    mock_s3.download_file.assert_called_once()


@patch("shared.s3_client.boto3")
def test_list_objects_paginates(mock_boto3: MagicMock) -> None:
    mock_s3 = MagicMock()
    mock_boto3.client.return_value = mock_s3

    paginator = MagicMock()
    paginator.paginate.return_value = [
        {"Contents": [{"Key": "a/1.bin"}, {"Key": "a/2.bin"}]},
        {"Contents": [{"Key": "a/3.bin"}]},
    ]
    mock_s3.get_paginator.return_value = paginator

    client = S3Client(bucket="test-bucket")
    keys = client.list_objects("a/")

    assert keys == ["a/1.bin", "a/2.bin", "a/3.bin"]
