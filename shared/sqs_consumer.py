"""SQS consumer base class for ML pipeline message processing."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

import boto3

from shared.logger import get_logger

logger = get_logger(__name__)


class SqsConsumer(ABC):
    """Base class for services that consume messages from an SQS queue.

    Subclasses must implement :meth:`process_message` to define
    domain-specific handling logic.
    """

    def __init__(self, queue_url: str, region: str = "us-east-1") -> None:
        """Initialise the consumer.

        Args:
            queue_url: Full URL of the SQS queue to poll.
            region: AWS region for the SQS client.
        """
        self.queue_url = queue_url
        self._client = boto3.client("sqs", region_name=region)

    @abstractmethod
    def process_message(self, body: dict[str, Any]) -> None:
        """Handle a single parsed message body.

        Args:
            body: The JSON-decoded message body.
        """

    def poll(self, max_messages: int = 1, wait_seconds: int = 20) -> None:
        """Long-poll the queue and process received messages.

        On success the message is deleted from the queue.  On failure the
        message is left in-flight so that it returns to the queue after the
        visibility timeout expires and eventually lands in the DLQ.

        Args:
            max_messages: Maximum number of messages to receive per poll.
            wait_seconds: Long-poll wait time in seconds.
        """
        response = self._client.receive_message(
            QueueUrl=self.queue_url,
            MaxNumberOfMessages=max_messages,
            WaitTimeSeconds=wait_seconds,
        )

        for message in response.get("Messages", []):
            message_id = message.get("MessageId", "unknown")
            try:
                body: dict[str, Any] = json.loads(message["Body"])
                self.process_message(body)
                self._client.delete_message(
                    QueueUrl=self.queue_url,
                    ReceiptHandle=message["ReceiptHandle"],
                )
                logger.info("Message processed successfully", extra={"message_id": message_id})
            except Exception:
                logger.error(
                    "Failed to process message; leaving on queue for retry/DLQ",
                    extra={"message_id": message_id},
                    exc_info=True,
                )

    def send_message(self, body: dict[str, Any]) -> None:
        """Publish a message to the queue.

        Args:
            body: Message payload that will be JSON-serialised.
        """
        self._client.send_message(
            QueueUrl=self.queue_url,
            MessageBody=json.dumps(body),
        )
