"""Thin HTTP client around the eToro Public API.

Responsibilities:
- Inject the partner ``x-api-key``, per-user ``x-user-key``, and a fresh
  ``x-request-id`` UUID v4 per request.
- Distinguish 401/403/413/414/429/5xx and raise the appropriate
  :mod:`errors` subclass.
- Apply *bounded* automatic retry only for 429 and 5xx (idempotent
  GETs only, never POSTs — trade execution has no idempotency key).
- Keep a single :class:`requests.Session` for connection reuse.

This client is deliberately **synchronous**. The bot's main loop runs
slowly enough (≥ 60s by default) that an asyncio overhaul would be a
maintenance tax for no measurable speedup.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import quote

import requests

from ..config import EtoroCredentials
from .errors import (
    EtoroApiError,
    EtoroAuthError,
    EtoroBadRequestError,
    EtoroPayloadTooLargeError,
    EtoroRateLimitError,
    EtoroServerError,
    EtoroTimeoutError,
)


_BASE_URL = "https://public-api.etoro.com/api/v1"
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_BACKOFF_SCHEDULE = (0.4, 1.5, 4.0)  # seconds per retry attempt
_TRADE_PATHS = ("/trading/execution/",)


@dataclass(frozen=True)
class HttpResult:
    status_code: int
    json: Any
    text: str


def _is_trade_execution(path: str) -> bool:
    return any(path.startswith(p) for p in _TRADE_PATHS)


class EtoroClient:
    """Bound to one set of credentials. Thread-unsafe by design."""

    def __init__(
        self,
        credentials: EtoroCredentials,
        *,
        request_timeout_seconds: int = 20,
        base_url: str = _BASE_URL,
        session: requests.Session | None = None,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
    ) -> None:
        self._creds = credentials
        self._timeout = request_timeout_seconds
        self._base_url = base_url.rstrip("/")
        self._session = session or requests.Session()
        self._logger = logger or logging.getLogger("etrader.etoro.client")

    # -- context manager --------------------------------------------------

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "EtoroClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- public verbs -----------------------------------------------------

    def get(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
        *,
        retries: int = 3,
    ) -> Any:
        return self._request("GET", path, params=params, retries=retries).json

    def post(
        self,
        path: str,
        json: Mapping[str, Any] | None = None,
        *,
        retries: int = 0,
    ) -> Any:
        # Trade-execution POSTs are at-most-once; never auto-retry them.
        if _is_trade_execution(path):
            retries = 0
        return self._request("POST", path, json_body=json, retries=retries).json

    # -- internals --------------------------------------------------------

    def _build_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-request-id": str(uuid.uuid4()),
            "x-api-key": self._creds.public_key,
            "x-user-key": self._creds.user_key,
        }

    def _build_url(self, path: str, params: Mapping[str, Any] | None) -> str:
        """Build a URL preserving literal commas in list-valued params.

        eToro's API rejects ``%2C`` in places where it expects a comma —
        e.g. ``instrumentIds=1,2,3`` — but ``requests`` (and most other
        URL builders) percent-encode the comma. We percent-encode every
        other unsafe char per RFC 3986 but keep ``,`` raw inside list
        values, which is the behavior eToro actually requires.
        """
        url = f"{self._base_url}{path}"
        if not params:
            return url
        parts: list[str] = []
        for key, value in params.items():
            if value is None:
                continue
            encoded_key = quote(str(key), safe="")
            if isinstance(value, (list, tuple)):
                encoded_value = ",".join(quote(str(item), safe="") for item in value)
            elif isinstance(value, str) and "," in value:
                # Caller already comma-joined — preserve the commas.
                encoded_value = ",".join(quote(piece, safe="") for piece in value.split(","))
            else:
                encoded_value = quote(str(value), safe="")
            parts.append(f"{encoded_key}={encoded_value}")
        if not parts:
            return url
        return f"{url}?{'&'.join(parts)}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json_body: Mapping[str, Any] | None = None,
        retries: int = 0,
    ) -> HttpResult:
        # eToro REJECTS percent-encoded commas in query params (returns 500).
        # `requests` encodes ',' → '%2C' by default, so we build the URL by
        # hand and pass params=None to bypass requests' own URL builder.
        url = self._build_url(path, params)

        attempt = 0
        last_exc: Exception | None = None
        while True:
            try:
                self._logger.debug("→ %s %s", method, path)
                response = self._session.request(
                    method,
                    url,
                    headers=self._build_headers(),
                    json=json_body,
                    timeout=self._timeout,
                )
            except requests.Timeout as exc:
                last_exc = exc
                if attempt < retries and method == "GET":
                    self._sleep_backoff(attempt)
                    attempt += 1
                    continue
                raise EtoroTimeoutError(str(exc)) from exc
            except requests.RequestException as exc:
                last_exc = exc
                # No automatic retry for trade execution
                if attempt < retries and method == "GET":
                    self._sleep_backoff(attempt)
                    attempt += 1
                    continue
                raise EtoroApiError(f"network error: {exc}") from exc

            status = response.status_code
            text = response.text or ""
            self._logger.debug("← %s %s [%d, %d bytes]", method, path, status, len(text))

            if 200 <= status < 300:
                try:
                    payload = response.json() if text else None
                except ValueError:
                    payload = None
                return HttpResult(status, payload, text)

            self._raise_or_continue(status, text, attempt, retries)
            attempt += 1

    def _raise_or_continue(self, status: int, text: str, attempt: int, retries: int) -> None:
        # Decide whether to retry or raise based on status & attempt count.
        if status in _RETRYABLE_STATUS and attempt < retries:
            self._sleep_backoff(attempt)
            return  # caller's outer loop will retry

        if status in (401, 403):
            raise EtoroAuthError(f"auth failed ({status})", status, text)
        if status in (413, 414):
            raise EtoroPayloadTooLargeError(f"payload too large ({status})", status, text)
        if status == 429:
            raise EtoroRateLimitError("rate limited", text)
        if 500 <= status < 600:
            raise EtoroServerError(f"server error ({status})", status, text)
        raise EtoroBadRequestError(f"request failed ({status})", status, text)

    @staticmethod
    def _sleep_backoff(attempt: int) -> None:
        delay = _BACKOFF_SCHEDULE[min(attempt, len(_BACKOFF_SCHEDULE) - 1)]
        time.sleep(delay)
