"""DynamoDB client for the ML model registry.

Provides CRUD operations against the ``foryou-model-registry`` DynamoDB table
used to track LoRA/DreamBooth model metadata, lifecycle status, and consent
enforcement state.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import boto3
from botocore.exceptions import ClientError

from shared.logger import get_logger

logger = get_logger(__name__)

TABLE_NAME = "foryou-model-registry"

# Valid model lifecycle transitions.
MODEL_STATUSES = ("training", "validating", "staging", "production", "revoked")


class ModelRegistryError(Exception):
    """Raised when a model registry operation fails."""


class DynamoDBClient:
    """Thin wrapper around boto3 DynamoDB operations for the model registry.

    The table schema expects:
    - Partition key: ``model_id`` (S)
    - GSI ``performer-index``: partition key ``performer_id`` (S), sort key ``created_at`` (S)
    """

    def __init__(self, region: str = "us-east-1", table_name: str = TABLE_NAME) -> None:
        """Initialise the DynamoDB client.

        Args:
            region: AWS region for the DynamoDB client.
            table_name: Name of the DynamoDB table.
        """
        self._table_name = table_name
        self._resource = boto3.resource("dynamodb", region_name=region)
        self._table = self._resource.Table(self._table_name)

    def put_item(self, item: dict[str, Any]) -> None:
        """Insert a new model record into the registry.

        Required fields: ``model_id``, ``performer_id``, ``version``, ``status``,
        ``consent_status``, ``supported_media_types``, ``s3_path``.

        Automatically sets ``created_at`` and ``updated_at`` if not present.

        Args:
            item: Model metadata dictionary.

        Raises:
            ModelRegistryError: If the put operation fails.
        """
        now = datetime.now(tz=UTC).isoformat()
        item.setdefault("created_at", now)
        item.setdefault("updated_at", now)

        required = ("model_id", "performer_id", "status", "consent_status", "s3_path")
        missing = [f for f in required if f not in item]
        if missing:
            raise ModelRegistryError(f"Missing required fields: {', '.join(missing)}")

        try:
            self._table.put_item(Item=item)
            logger.info(
                "Registered model in registry",
                extra={"model_id": item["model_id"], "performer_id": item["performer_id"]},
            )
        except ClientError as exc:
            logger.error(
                "Failed to register model",
                extra={"model_id": item.get("model_id"), "error": str(exc)},
            )
            raise ModelRegistryError(f"Failed to put item: {exc}") from exc

    def get_item(self, model_id: str) -> dict[str, Any] | None:
        """Retrieve a model record by its unique identifier.

        Args:
            model_id: The model's unique identifier (partition key).

        Returns:
            The model metadata dict, or ``None`` if not found.

        Raises:
            ModelRegistryError: If the get operation fails.
        """
        try:
            response = self._table.get_item(Key={"model_id": model_id})
            item = response.get("Item")
            if item is None:
                logger.info("Model not found", extra={"model_id": model_id})
            return item
        except ClientError as exc:
            logger.error(
                "Failed to get model",
                extra={"model_id": model_id, "error": str(exc)},
            )
            raise ModelRegistryError(f"Failed to get item: {exc}") from exc

    def update_item(
        self,
        model_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        """Update specific attributes on an existing model record.

        Automatically bumps ``updated_at``.

        Args:
            model_id: The model's unique identifier.
            updates: Key-value pairs to update (e.g. ``{"status": "production"}``).

        Returns:
            The full updated item attributes.

        Raises:
            ModelRegistryError: If the update operation fails.
        """
        updates["updated_at"] = datetime.now(tz=UTC).isoformat()

        # Build UpdateExpression dynamically.
        expr_parts: list[str] = []
        expr_names: dict[str, str] = {}
        expr_values: dict[str, Any] = {}

        for i, (key, value) in enumerate(updates.items()):
            alias_name = f"#k{i}"
            alias_value = f":v{i}"
            expr_parts.append(f"{alias_name} = {alias_value}")
            expr_names[alias_name] = key
            expr_values[alias_value] = value

        update_expression = "SET " + ", ".join(expr_parts)

        try:
            response = self._table.update_item(
                Key={"model_id": model_id},
                UpdateExpression=update_expression,
                ExpressionAttributeNames=expr_names,
                ExpressionAttributeValues=expr_values,
                ReturnValues="ALL_NEW",
            )
            logger.info(
                "Updated model in registry",
                extra={"model_id": model_id, "updated_fields": list(updates.keys())},
            )
            return response["Attributes"]
        except ClientError as exc:
            logger.error(
                "Failed to update model",
                extra={"model_id": model_id, "error": str(exc)},
            )
            raise ModelRegistryError(f"Failed to update item: {exc}") from exc

    def query_by_performer(
        self,
        performer_id: str,
        status: str | None = None,
        consent_status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query all models belonging to a performer, optionally filtered.

        Uses the ``performer-index`` GSI. Results are sorted by ``created_at``
        descending (newest first).

        Args:
            performer_id: The performer's UUID.
            status: Optional lifecycle status filter (e.g. ``"production"``).
            consent_status: Optional consent status filter (e.g. ``"active"``).

        Returns:
            A list of matching model metadata dicts.

        Raises:
            ModelRegistryError: If the query operation fails.
        """
        key_condition = "performer_id = :pid"
        expr_values: dict[str, Any] = {":pid": performer_id}
        filter_parts: list[str] = []

        if status is not None:
            filter_parts.append("#st = :status")
            expr_values[":status"] = status

        if consent_status is not None:
            filter_parts.append("consent_status = :cs")
            expr_values[":cs"] = consent_status

        kwargs: dict[str, Any] = {
            "IndexName": "performer-index",
            "KeyConditionExpression": key_condition,
            "ExpressionAttributeValues": expr_values,
            "ScanIndexForward": False,
        }

        # 'status' is a DynamoDB reserved word, so we alias it.
        if status is not None:
            kwargs["ExpressionAttributeNames"] = {"#st": "status"}

        if filter_parts:
            kwargs["FilterExpression"] = " AND ".join(filter_parts)

        try:
            items: list[dict[str, Any]] = []
            while True:
                response = self._table.query(**kwargs)
                items.extend(response.get("Items", []))
                last_key = response.get("LastEvaluatedKey")
                if last_key is None:
                    break
                kwargs["ExclusiveStartKey"] = last_key

            logger.info(
                "Queried models for performer",
                extra={
                    "performer_id": performer_id,
                    "result_count": len(items),
                    "status_filter": status,
                    "consent_filter": consent_status,
                },
            )
            return items
        except ClientError as exc:
            logger.error(
                "Failed to query models for performer",
                extra={"performer_id": performer_id, "error": str(exc)},
            )
            raise ModelRegistryError(f"Failed to query: {exc}") from exc
