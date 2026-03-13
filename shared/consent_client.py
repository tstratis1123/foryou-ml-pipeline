"""HTTP client for the Consent Service consent check endpoint."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from shared.logger import get_logger

logger = get_logger(__name__)


class ConsentDeniedError(Exception):
    """Raised when consent check fails (not found, revoked, or media type unsupported)."""


class ConsentCheckUnavailableError(Exception):
    """Raised when the consent service is unreachable."""


class ConsentClient:
    """HTTP client for checking performer consent before pipeline operations.

    Per Critical Invariant #1: no generation, no training, no content
    delivery without a verified consent check.  No retries — if the
    service is down the job must fail.
    """

    def __init__(self, base_url: str, timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def check_consent(self, performer_id: str, media_type: str) -> dict[str, Any]:
        """Verify a performer has active consent for the given media type.

        Args:
            performer_id: UUID of the performer.
            media_type: Either ``"image"`` or ``"video"``.

        Returns:
            The consent check response body.

        Raises:
            ConsentDeniedError: Consent is not active or media type
                not in performer's supported types.
            ConsentCheckUnavailableError: Service cannot be reached.
        """
        url = f"{self.base_url}/consent/check"
        payload = json.dumps({"performerId": performer_id, "mediaType": media_type}).encode("utf-8")

        logger.info("Checking consent", extra={"performer_id": performer_id, "media_type": media_type})

        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            status = exc.code
            if status in (403, 404, 422):
                body = exc.read().decode("utf-8", errors="replace")
                logger.warning("Consent denied", extra={"performer_id": performer_id, "status": status, "body": body})
                raise ConsentDeniedError(f"Consent denied for performer {performer_id}: HTTP {status}") from exc
            logger.error("Consent service error", extra={"performer_id": performer_id, "status": status})
            raise ConsentCheckUnavailableError(f"Consent service returned HTTP {status}") from exc
        except (urllib.error.URLError, OSError) as exc:
            logger.error("Consent service unreachable", extra={"performer_id": performer_id, "error": str(exc)})
            raise ConsentCheckUnavailableError(f"Cannot reach consent service: {exc}") from exc

        has_consent = result.get("hasConsent", False)
        if not has_consent:
            logger.warning("Consent not active", extra={"performer_id": performer_id, "media_type": media_type})
            raise ConsentDeniedError(f"No active consent for performer {performer_id} media_type={media_type}")

        logger.info("Consent verified", extra={"performer_id": performer_id, "media_type": media_type})
        return result
