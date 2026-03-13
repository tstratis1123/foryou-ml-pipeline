"""Unit tests for the ConsentClient."""

from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from shared.consent_client import ConsentCheckUnavailableError, ConsentClient, ConsentDeniedError


def _mock_response(body: dict, status: int = 200) -> MagicMock:
    """Create a mock urllib response."""
    resp = MagicMock()
    resp.read.return_value = json.dumps(body).encode("utf-8")
    resp.status = status
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _http_error(status: int, body: str = "") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="http://test/consent/check",
        code=status,
        msg="error",
        hdrs=None,  # type: ignore[arg-type]
        fp=BytesIO(body.encode("utf-8")),
    )


class TestConsentClientSuccess:
    @patch("shared.consent_client.urllib.request.urlopen")
    def test_check_consent_success(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response({"hasConsent": True, "performerId": "p1"})

        client = ConsentClient(base_url="http://consent:3002")
        result = client.check_consent("p1", "image")

        assert result["hasConsent"] is True

    @patch("shared.consent_client.urllib.request.urlopen")
    def test_check_consent_has_consent_false(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response({"hasConsent": False})

        client = ConsentClient(base_url="http://consent:3002")
        with pytest.raises(ConsentDeniedError, match="No active consent"):
            client.check_consent("p1", "image")


class TestConsentClientDenied:
    @patch("shared.consent_client.urllib.request.urlopen")
    def test_consent_revoked_403(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = _http_error(403, "CONSENT_REVOKED")

        client = ConsentClient(base_url="http://consent:3002")
        with pytest.raises(ConsentDeniedError, match="HTTP 403"):
            client.check_consent("p1", "image")

    @patch("shared.consent_client.urllib.request.urlopen")
    def test_consent_not_found_404(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = _http_error(404)

        client = ConsentClient(base_url="http://consent:3002")
        with pytest.raises(ConsentDeniedError, match="HTTP 404"):
            client.check_consent("p1", "video")

    @patch("shared.consent_client.urllib.request.urlopen")
    def test_consent_media_type_unsupported_422(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = _http_error(422, "MEDIA_TYPE_NOT_SUPPORTED")

        client = ConsentClient(base_url="http://consent:3002")
        with pytest.raises(ConsentDeniedError):
            client.check_consent("p1", "video")


class TestConsentClientUnavailable:
    @patch("shared.consent_client.urllib.request.urlopen")
    def test_service_error_503(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = _http_error(503)

        client = ConsentClient(base_url="http://consent:3002")
        with pytest.raises(ConsentCheckUnavailableError, match="HTTP 503"):
            client.check_consent("p1", "image")

    @patch("shared.consent_client.urllib.request.urlopen")
    def test_network_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        client = ConsentClient(base_url="http://consent:3002")
        with pytest.raises(ConsentCheckUnavailableError, match="Cannot reach"):
            client.check_consent("p1", "image")

    @patch("shared.consent_client.urllib.request.urlopen")
    def test_timeout(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = urllib.error.URLError(OSError("timed out"))

        client = ConsentClient(base_url="http://consent:3002")
        with pytest.raises(ConsentCheckUnavailableError):
            client.check_consent("p1", "image")
