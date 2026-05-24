"""eToro Public API client + endpoint wrappers.

Submodules:
- :mod:`errors` — typed error hierarchy (rate-limit, session-expired, etc.).
- :mod:`client` — HTTP session with retries, headers, auth.
- :mod:`market_data` — search, instruments, rates, candles.
- :mod:`trading` — open / close orders, PnL endpoint.
- :mod:`identity` — `/api/v1/me`.
- :mod:`watchlists` — curated lists, market recommendations.
- :mod:`instrument_cache` — JSON-persisted symbol↔instrumentID map.
"""

from .errors import (
    EtoroApiError,
    EtoroAuthError,
    EtoroBadRequestError,
    EtoroPayloadTooLargeError,
    EtoroRateLimitError,
    EtoroServerError,
    EtoroTimeoutError,
)

__all__ = [
    "EtoroApiError",
    "EtoroAuthError",
    "EtoroBadRequestError",
    "EtoroPayloadTooLargeError",
    "EtoroRateLimitError",
    "EtoroServerError",
    "EtoroTimeoutError",
]
