"""Structured logging configuration for the ML pipeline."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter for ML pipeline services.

    Outputs each log record as a single JSON line containing timestamp,
    level, logger name, message, and any extra fields attached to the record.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string.

        Args:
            record: The log record to format.

        Returns:
            A single-line JSON string representing the log entry.
        """
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Merge extra fields added via the `extra` kwarg on log calls.
        # Standard LogRecord attributes are excluded to keep output clean.
        _standard_attrs = {
            "name",
            "msg",
            "args",
            "created",
            "relativeCreated",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "pathname",
            "filename",
            "module",
            "levelno",
            "levelname",
            "msecs",
            "processName",
            "process",
            "threadName",
            "thread",
            "taskName",
            "message",
        }
        for key, value in record.__dict__.items():
            if key not in _standard_attrs and not key.startswith("_"):
                log_entry[key] = value

        if record.exc_info and record.exc_info[1] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a structured JSON logger for the given module name.

    If the logger already has handlers (e.g. from a previous call), it is
    returned as-is to avoid duplicate output.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        level: Minimum log level. Defaults to ``logging.INFO``.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(StructuredFormatter())
    logger.addHandler(handler)

    return logger
