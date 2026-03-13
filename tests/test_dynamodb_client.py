"""Unit tests for DynamoDB model registry client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from shared.dynamodb_client import DynamoDBClient, ModelRegistryError


@patch("shared.dynamodb_client.boto3")
def _make_client(mock_boto3: MagicMock) -> tuple[DynamoDBClient, MagicMock]:
    mock_table = MagicMock()
    mock_resource = MagicMock()
    mock_resource.Table.return_value = mock_table
    mock_boto3.resource.return_value = mock_resource
    client = DynamoDBClient(region="us-east-1")
    return client, mock_table


def _valid_item() -> dict:
    return {
        "model_id": "m1",
        "performer_id": "p1",
        "status": "training",
        "consent_status": "active",
        "s3_path": "models/p1/m1/",
    }


class TestPutItem:
    def test_put_item_success(self) -> None:
        client, mock_table = _make_client()

        client.put_item(_valid_item())

        mock_table.put_item.assert_called_once()
        item = mock_table.put_item.call_args[1]["Item"]
        assert item["model_id"] == "m1"
        assert item["performer_id"] == "p1"
        assert item["status"] == "training"
        assert "created_at" in item
        assert "updated_at" in item

    def test_put_item_missing_required_field(self) -> None:
        client, _ = _make_client()

        with pytest.raises(ModelRegistryError, match="Missing required"):
            client.put_item({"model_id": "m1", "performer_id": "p1"})


class TestGetItem:
    def test_get_item_found(self) -> None:
        client, mock_table = _make_client()
        mock_table.get_item.return_value = {"Item": {"model_id": "m1", "status": "active"}}

        result = client.get_item("m1")

        assert result is not None
        assert result["model_id"] == "m1"

    def test_get_item_not_found(self) -> None:
        client, mock_table = _make_client()
        mock_table.get_item.return_value = {}

        result = client.get_item("missing")

        assert result is None


class TestQueryByPerformer:
    def test_query_returns_items(self) -> None:
        client, mock_table = _make_client()
        mock_table.query.return_value = {"Items": [{"model_id": "m1"}, {"model_id": "m2"}]}

        results = client.query_by_performer("p1")

        assert len(results) == 2
        mock_table.query.assert_called_once()
