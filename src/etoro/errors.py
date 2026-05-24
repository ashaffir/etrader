"""Typed error hierarchy for eToro API responses.

Each subclass carries a stable ``name`` so callers can branch on it
across module boundaries (per the *building-etoro-api-client* skill).
"""

from __future__ import annotations


class EtoroApiError(Exception):
    """Base class for any HTTP error coming from the Public API."""

    def __init__(self, message: str, status_code: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.name = "EtoroApiError"


class EtoroAuthError(EtoroApiError):
    """401 / 403 — invalid or insufficient credentials."""

    def __init__(self, message: str = "eToro auth failed", status_code: int = 401, body: str = "") -> None:
        super().__init__(message, status_code, body)
        self.name = "EtoroAuthError"


class EtoroRateLimitError(EtoroApiError):
    """429 — backoff and retry at the *same* payload size."""

    def __init__(self, message: str = "eToro rate limit", body: str = "") -> None:
        super().__init__(message, 429, body)
        self.name = "EtoroRateLimitError"


class EtoroPayloadTooLargeError(EtoroApiError):
    """413 / 414 — halve the payload and retry."""

    def __init__(self, message: str = "eToro payload too large", status_code: int = 413, body: str = "") -> None:
        super().__init__(message, status_code, body)
        self.name = "EtoroPayloadTooLargeError"


class EtoroBadRequestError(EtoroApiError):
    """4xx (other than 401/403/413/414/429) — request shape is wrong."""

    def __init__(self, message: str, status_code: int, body: str = "") -> None:
        super().__init__(message, status_code, body)
        self.name = "EtoroBadRequestError"


class EtoroServerError(EtoroApiError):
    """5xx — eToro-side issue, retry with backoff."""

    def __init__(self, message: str = "eToro server error", status_code: int = 500, body: str = "") -> None:
        super().__init__(message, status_code, body)
        self.name = "EtoroServerError"


class EtoroTimeoutError(EtoroApiError):
    """Connection timeout / no response — ambiguous, do not auto-retry trades."""

    def __init__(self, message: str = "eToro request timed out", body: str = "") -> None:
        super().__init__(message, None, body)
        self.name = "EtoroTimeoutError"
