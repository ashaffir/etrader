"""HTTP client for the trading bot's internal control API.

Just enough to wrap the routes registered in
:mod:`src.control.handlers`. We use ``requests`` (already a project
dependency) for connection reuse and timeout support.

All errors get wrapped into :class:`ControlAPIError` so the Telegram
command layer can render a single user-friendly message regardless of
whether the failure was a network blip, a 4xx, or a 5xx.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import requests


class ControlAPIError(Exception):
    """Raised when the control API returns a non-2xx or is unreachable."""


@dataclass
class ControlAPIClient:
    base_url: str
    token: str
    timeout_seconds: float = 30.0
    session: requests.Session | None = None
    logger: logging.Logger | None = None

    def __post_init__(self) -> None:
        if not self.token:
            raise ValueError("INTERNAL_API_TOKEN is required")
        if self.session is None:
            self.session = requests.Session()
        if self.logger is None:
            self.logger = logging.getLogger("etrader.telegram.control_client")
        self.base_url = self.base_url.rstrip("/")

    # -- low-level ------------------------------------------------------

    def _request(self, method: str, path: str, *, body: dict[str, Any] | None = None) -> Any:
        assert self.session is not None
        url = f"{self.base_url}{path}"
        headers = {"Authorization": f"Bearer {self.token}", "Accept": "application/json"}
        try:
            resp = self.session.request(
                method.upper(),
                url,
                headers=headers,
                json=body,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise ControlAPIError(f"control API unreachable at {url}: {exc}") from exc

        try:
            payload = resp.json() if resp.content else None
        except ValueError:
            payload = None

        if resp.status_code >= 400:
            err = (payload or {}).get("error") if isinstance(payload, dict) else None
            raise ControlAPIError(err or f"HTTP {resp.status_code}: {resp.text[:200]}")
        return payload

    # -- routes ---------------------------------------------------------

    def ping(self) -> dict[str, Any]:
        return self._request("GET", "/ping")

    def status(self) -> dict[str, Any]:
        return self._request("GET", "/status")

    def portfolio(self) -> dict[str, Any]:
        return self._request("GET", "/portfolio")

    def universe(self) -> dict[str, Any]:
        return self._request("GET", "/universe")

    def news(self, *, limit: int = 25) -> dict[str, Any]:
        limit = max(1, min(int(limit), 200))
        return self._request("GET", f"/news?limit={limit}")

    def fundamentals(self, *, symbol: str | None = None) -> dict[str, Any]:
        if symbol:
            from urllib.parse import quote
            return self._request("GET", f"/fundamentals?symbol={quote(symbol.strip())}")
        return self._request("GET", "/fundamentals")

    def history(self, *, limit: int = 20) -> dict[str, Any]:
        limit = max(1, min(int(limit), 200))
        return self._request("GET", f"/history?limit={limit}")

    def pause(self, *, reason: str | None = None) -> dict[str, Any]:
        return self._request("POST", "/pause", body={"reason": reason})

    def resume(self) -> dict[str, Any]:
        return self._request("POST", "/resume")

    def panic(self, *, scope: str = "all", reason: str | None = None) -> dict[str, Any]:
        return self._request("POST", "/panic", body={"scope": scope, "reason": reason})

    def get_guardrails(self) -> dict[str, Any]:
        return self._request("GET", "/config/guardrails")

    def set_guardrail(self, key: str, value: Any) -> dict[str, Any]:
        return self._request("POST", "/config/guardrails", body={"key": key, "value": value})

    def ask(self, question: str) -> dict[str, Any]:
        return self._request("POST", "/ask", body={"question": question})

    def strategy_signals(self) -> dict[str, Any]:
        return self._request("GET", "/strategy/signals")

    def news_channels(self) -> dict[str, Any]:
        return self._request("GET", "/news/channels")

    def news_channels_test(
        self, *, only: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if only:
            body["only"] = list(only)
        return self._request("POST", "/news/channels/test", body=body or None)

    # -- alerts ---------------------------------------------------------

    def alert_types(self) -> dict[str, Any]:
        return self._request("GET", "/alerts/types")

    def alert_subscriptions(self, chat_id: int) -> dict[str, Any]:
        return self._request("GET", f"/alerts/subscriptions?chat_id={int(chat_id)}")

    def set_alert_subscription(
        self, chat_id: int, type_str: str, enabled: bool,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/alerts/subscriptions",
            body={"chat_id": int(chat_id), "type": type_str, "enabled": bool(enabled)},
        )

    def toggle_alert_subscription(
        self, chat_id: int, type_str: str,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/alerts/subscriptions",
            body={"chat_id": int(chat_id), "type": type_str, "toggle": True},
        )

    def alert_pending(self, chat_id: int, *, limit: int = 50) -> dict[str, Any]:
        limit = max(1, min(int(limit), 200))
        return self._request(
            "GET",
            f"/alerts/pending?chat_id={int(chat_id)}&limit={limit}",
        )
