"""Unit tests for SQS consumer error handling."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

from shared.sqs_consumer import SqsConsumer


class _StubConsumer(SqsConsumer):
    """Concrete subclass for testing."""

    def __init__(self, queue_url: str = "https://sqs.us-east-1.amazonaws.com/123/test") -> None:
        # Skip real boto3 client creation
        self.queue_url = queue_url
        self._client = MagicMock()
        self.processed: list[dict[str, Any]] = []

    def process_message(self, body: dict[str, Any]) -> None:
        self.processed.append(body)


def _make_message(body: dict[str, Any], message_id: str = "msg-1") -> dict[str, Any]:
    return {
        "MessageId": message_id,
        "ReceiptHandle": f"handle-{message_id}",
        "Body": json.dumps(body),
    }


def test_poll_success_deletes_message() -> None:
    consumer = _StubConsumer()
    msg = _make_message({"job_id": "j1"})
    consumer._client.receive_message.return_value = {"Messages": [msg]}

    consumer.poll()

    assert consumer.processed == [{"job_id": "j1"}]
    consumer._client.delete_message.assert_called_once_with(
        QueueUrl=consumer.queue_url,
        ReceiptHandle="handle-msg-1",
    )


def test_poll_failure_does_not_delete_message() -> None:
    consumer = _StubConsumer()
    consumer.process_message = MagicMock(side_effect=RuntimeError("boom"))  # type: ignore[assignment]
    msg = _make_message({"job_id": "j1"})
    consumer._client.receive_message.return_value = {"Messages": [msg]}

    # Should not raise — error is caught internally
    consumer.poll()

    consumer._client.delete_message.assert_not_called()


def test_poll_continues_after_failure() -> None:
    call_count = 0

    class _FailFirstConsumer(_StubConsumer):
        def process_message(self, body: dict[str, Any]) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("first fails")
            self.processed.append(body)

    consumer = _FailFirstConsumer()
    msg1 = _make_message({"job_id": "j1"}, "msg-1")
    msg2 = _make_message({"job_id": "j2"}, "msg-2")
    consumer._client.receive_message.return_value = {"Messages": [msg1, msg2]}

    consumer.poll(max_messages=2)

    # Second message processed and deleted; first not deleted
    assert consumer.processed == [{"job_id": "j2"}]
    consumer._client.delete_message.assert_called_once_with(
        QueueUrl=consumer.queue_url,
        ReceiptHandle="handle-msg-2",
    )


def test_poll_no_messages() -> None:
    consumer = _StubConsumer()
    consumer._client.receive_message.return_value = {}

    consumer.poll()

    assert consumer.processed == []
    consumer._client.delete_message.assert_not_called()


def test_send_message() -> None:
    consumer = _StubConsumer()
    payload = {"event": "training.completed", "model_id": "m1"}

    consumer.send_message(payload)

    consumer._client.send_message.assert_called_once_with(
        QueueUrl=consumer.queue_url,
        MessageBody=json.dumps(payload),
    )
