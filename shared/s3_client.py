"""S3 client helper for uploading and downloading ML artifacts."""

from __future__ import annotations

import boto3


class S3Client:
    """Thin wrapper around boto3 S3 operations used by ML pipelines."""

    def __init__(self, bucket: str, region: str = "us-east-1") -> None:
        """Initialise the S3 client.

        Args:
            bucket: Default S3 bucket name for all operations.
            region: AWS region for the S3 client.
        """
        self.bucket = bucket
        self._client = boto3.client("s3", region_name=region)

    def upload_file(self, local_path: str, s3_key: str) -> None:
        """Upload a local file to S3.

        Args:
            local_path: Path to the file on the local filesystem.
            s3_key: Destination key within the configured bucket.
        """
        self._client.upload_file(local_path, self.bucket, s3_key)

    def download_file(self, s3_key: str, local_path: str) -> None:
        """Download a file from S3 to the local filesystem.

        Args:
            s3_key: Source key within the configured bucket.
            local_path: Destination path on the local filesystem.
        """
        self._client.download_file(self.bucket, s3_key, local_path)

    def list_objects(self, prefix: str) -> list[str]:
        """List object keys under a given S3 prefix.

        Args:
            prefix: S3 key prefix to list.

        Returns:
            A list of matching S3 object keys.
        """
        paginator = self._client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys
