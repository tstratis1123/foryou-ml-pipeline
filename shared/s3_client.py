"""S3 client helper for uploading and downloading ML artifacts."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from shared.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

_NON_RETRYABLE_CODES = frozenset({"NoSuchBucket", "NoSuchKey", "AccessDenied", "403", "404"})


class S3Client:
    """Thin wrapper around boto3 S3 operations used by ML pipelines."""

    def __init__(self, bucket: str, region: str = "us-east-1") -> None:
        """Initialise the S3 client.

        Args:
            bucket: Default S3 bucket name for all operations.
            region: AWS region for the S3 client.
        """
        self.bucket = bucket
        self._client = boto3.client(
            "s3",
            region_name=region,
            config=BotoConfig(
                connect_timeout=10,
                read_timeout=30,
                retries={"max_attempts": 0},
            ),
        )

    def _retry(self, operation: Callable[[], T], *, max_attempts: int = 3, base_delay: float = 1.0) -> T:
        """Execute *operation* with exponential backoff on transient errors.

        Non-retryable errors (404, 403, etc.) are raised immediately.
        """
        for attempt in range(1, max_attempts + 1):
            try:
                return operation()
            except ClientError as exc:
                error_code = exc.response["Error"]["Code"]
                if error_code in _NON_RETRYABLE_CODES:
                    logger.error("S3 non-retryable error", extra={"error_code": error_code})
                    raise
                if attempt == max_attempts:
                    logger.error(
                        "S3 operation failed after retries",
                        extra={"error_code": error_code, "attempts": max_attempts},
                    )
                    raise
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "S3 transient error, retrying",
                    extra={"error_code": error_code, "attempt": attempt, "delay": delay},
                )
                time.sleep(delay)
        # Unreachable, but keeps mypy happy
        raise RuntimeError("retry loop exited unexpectedly")

    def upload_file(self, local_path: str, s3_key: str) -> None:
        """Upload a local file to S3 with retries.

        Args:
            local_path: Path to the file on the local filesystem.
            s3_key: Destination key within the configured bucket.
        """
        logger.info("Uploading to S3", extra={"s3_key": s3_key})
        self._retry(lambda: self._client.upload_file(local_path, self.bucket, s3_key))
        logger.info("Upload complete", extra={"s3_key": s3_key})

    def download_file(self, s3_key: str, local_path: str) -> None:
        """Download a file from S3 to the local filesystem with retries.

        Args:
            s3_key: Source key within the configured bucket.
            local_path: Destination path on the local filesystem.
        """
        logger.info("Downloading from S3", extra={"s3_key": s3_key})
        self._retry(lambda: self._client.download_file(self.bucket, s3_key, local_path))
        logger.info("Download complete", extra={"s3_key": s3_key})

    def list_objects(self, prefix: str) -> list[str]:
        """List object keys under a given S3 prefix with retries.

        Args:
            prefix: S3 key prefix to list.

        Returns:
            A list of matching S3 object keys.
        """
        logger.info("Listing S3 objects", extra={"prefix": prefix})

        def _paginate() -> list[str]:
            paginator = self._client.get_paginator("list_objects_v2")
            keys: list[str] = []
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    keys.append(obj["Key"])
            return keys

        result = self._retry(_paginate)
        logger.info("Listed %d objects", len(result), extra={"prefix": prefix})
        return result
