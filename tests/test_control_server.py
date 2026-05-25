"""Integration tests for the internal HTTP control server.

Boots a real :class:`ThreadingHTTPServer` on a random localhost port,
fires HTTP requests at it via stdlib ``urllib``, and asserts the
auth + routing + happy-path responses. We use the same
``BotController`` test scaffolding as ``test_controller.py``.
"""

import json
import logging
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from src.config import (
    AiConfig,
    AlertingConfig,
    AppConfig,
    AzureCredentials,
    ControlServiceConfig,
    EtoroCredentials,
    FundamentalsConfig,
    GuardrailsConfig,
    LoggingConfig,
    NewsConfig,
    OperationsConfig,
    StrategyConfig,
    ToolsConfig,
    UniverseConfig,
)
from src.alerts import AlertHub, AlertSubscriptions, safety_only_default
from src.control.controller import BotController
from src.control.server import ControlHTTPServer
from src.persistence import StatePersistence
from src.state import BotState
from src.telemetry import TelemetryStore
from src.trade_history import TradeHistoryLog


class _StubEtoro:
    def get(self, path, params=None, retries=0):  # noqa: ARG002
        return {"clientPortfolio": {"credit": 1000.0, "positions": []}}

    def post(self, path, json=None, retries=0):  # noqa: ARG002
        return {}


def _build_app_cfg() -> AppConfig:
    return AppConfig(
        trading_mode="paper",
        guardrails=GuardrailsConfig(),
        operations=OperationsConfig(trade_spacing_seconds=0),
        universe=UniverseConfig(),
        news=NewsConfig(enabled=False),
        fundamentals=FundamentalsConfig(enabled=False),
        strategy=StrategyConfig(),
        ai=AiConfig(enabled=False),
        tools=ToolsConfig(enabled=False),
        logging=LoggingConfig(),
        etoro=EtoroCredentials(public_key="x", user_key="y", is_real=False, allow_real=False),
        azure=AzureCredentials(endpoint=None, api_key=None, deployment=None),
        control=ControlServiceConfig(internal_api_token="secret-token-test"),
        alerting=AlertingConfig(allowed_chat_ids=(101, 202)),
    )


def _http_get(url: str, *, token: str | None) -> tuple[int, dict | None]:
    req = urllib.request.Request(url, method="GET")
    if token is not None:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "null")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8") or "{}"
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, None


def _http_post(url: str, *, token: str, body: dict | None) -> tuple[int, dict | None]:
    data = json.dumps(body or {}).encode("utf-8")
    req = urllib.request.Request(url, method="POST", data=data)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "null")
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8") or "{}"
        try:
            return exc.code, json.loads(body_text)
        except json.JSONDecodeError:
            return exc.code, None


class ControlServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp = Path(self._tmp.name)

        log = logging.getLogger("test.control.server")
        log.addHandler(logging.NullHandler())

        subs = AlertSubscriptions(
            tmp / "subs.json", default_set=safety_only_default(),
        )
        self.alerts = AlertHub(allowed_chat_ids=(101, 202), subscriptions=subs)
        self.controller = BotController(
            cfg=_build_app_cfg(),
            state=BotState(),
            etoro=_StubEtoro(),
            ai_client=None,
            telemetry=TelemetryStore(),
            history=TradeHistoryLog(tmp / "history.jsonl"),
            persistence=StatePersistence(tmp / "state.json"),
            alerts=self.alerts,
            logger=log,
        )
        # Bind to an OS-assigned port (port=0) for parallel test safety.
        self.server = ControlHTTPServer(
            host="127.0.0.1",
            port=0,
            bearer_token="secret-token-test",
            controller=self.controller,
            logger=log,
        )
        self.server.start()
        # Discover the actual bound port.
        assert self.server._httpd is not None  # noqa: SLF001
        self.port = self.server._httpd.server_address[1]  # noqa: SLF001
        self.base = f"http://127.0.0.1:{self.port}"
        self.addCleanup(self.server.stop)

    def test_unauth_request_rejected(self) -> None:
        status, body = _http_get(f"{self.base}/status", token=None)
        self.assertEqual(status, 401)
        assert body is not None
        self.assertIn("unauthorized", body.get("error", ""))

    def test_wrong_token_rejected(self) -> None:
        status, _ = _http_get(f"{self.base}/status", token="wrong")
        self.assertEqual(status, 401)

    def test_status_ok(self) -> None:
        status, body = _http_get(f"{self.base}/status", token="secret-token-test")
        self.assertEqual(status, 200)
        assert body is not None
        self.assertIn("trading_mode", body)
        self.assertEqual(body["trading_mode"], "paper")

    def test_pause_then_status_reflects_paused(self) -> None:
        st, _ = _http_post(f"{self.base}/pause", token="secret-token-test", body={"reason": "test"})
        self.assertEqual(st, 200)
        st2, body2 = _http_get(f"{self.base}/status", token="secret-token-test")
        self.assertEqual(st2, 200)
        assert body2 is not None
        self.assertTrue(body2["paused"])

    def test_unknown_route_404(self) -> None:
        st, body = _http_get(f"{self.base}/does-not-exist", token="secret-token-test")
        self.assertEqual(st, 404)
        assert body is not None
        self.assertIn("unknown route", body.get("error", ""))

    def test_set_guardrail_round_trip(self) -> None:
        st, body = _http_post(
            f"{self.base}/config/guardrails",
            token="secret-token-test",
            body={"key": "max_per_trade_usd", "value": 123},
        )
        self.assertEqual(st, 200)
        assert body is not None
        self.assertEqual(body["current"], 123.0)

        st2, body2 = _http_get(f"{self.base}/config/guardrails", token="secret-token-test")
        self.assertEqual(st2, 200)
        assert body2 is not None
        self.assertEqual(body2["guardrails"]["max_per_trade_usd"], 123.0)

    def test_unknown_guardrail_returns_400(self) -> None:
        st, _ = _http_post(
            f"{self.base}/config/guardrails",
            token="secret-token-test",
            body={"key": "bogus", "value": 1},
        )
        self.assertEqual(st, 400)

    # ----- /alerts endpoints --------------------------------------------------

    def test_alerts_types_lists_known_types(self) -> None:
        st, body = _http_get(
            f"{self.base}/alerts/types", token="secret-token-test",
        )
        self.assertEqual(st, 200)
        assert body is not None
        types = [t["type"] for t in body["types"]]
        self.assertIn("trade_opened", types)
        self.assertIn("panic_close", types)

    def test_alerts_subscriptions_get_seeds_default(self) -> None:
        st, body = _http_get(
            f"{self.base}/alerts/subscriptions?chat_id=101",
            token="secret-token-test",
        )
        self.assertEqual(st, 200)
        assert body is not None
        self.assertEqual(body["chat_id"], 101)
        self.assertIn("panic_close", body["enabled"])

    def test_alerts_subscriptions_set_persists(self) -> None:
        st, body = _http_post(
            f"{self.base}/alerts/subscriptions",
            token="secret-token-test",
            body={"chat_id": 101, "type": "trade_opened", "enabled": True},
        )
        self.assertEqual(st, 200)
        assert body is not None
        self.assertTrue(body["enabled"])
        self.assertIn("trade_opened", body["all_enabled"])

        # Verify GET reflects the change
        _, fresh = _http_get(
            f"{self.base}/alerts/subscriptions?chat_id=101",
            token="secret-token-test",
        )
        assert fresh is not None
        self.assertIn("trade_opened", fresh["enabled"])

    def test_alerts_subscriptions_toggle(self) -> None:
        # Default has panic_close ON; toggle should turn it OFF.
        st, body = _http_post(
            f"{self.base}/alerts/subscriptions",
            token="secret-token-test",
            body={"chat_id": 101, "type": "panic_close", "toggle": True},
        )
        self.assertEqual(st, 200)
        assert body is not None
        self.assertFalse(body["enabled"])

    def test_alerts_subscriptions_missing_chat_id_400(self) -> None:
        st, _ = _http_get(
            f"{self.base}/alerts/subscriptions",
            token="secret-token-test",
        )
        self.assertEqual(st, 400)

    # ----- /news/channels endpoints ------------------------------------------

    def test_news_channels_returns_payload_when_pipeline_absent(self) -> None:
        # No news pipeline wired in this controller -> empty channels list,
        # but the endpoint must still 200 with the documented shape.
        st, body = _http_get(
            f"{self.base}/news/channels", token="secret-token-test",
        )
        self.assertEqual(st, 200)
        assert body is not None
        self.assertIn("channels", body)
        self.assertIn("pipeline_enabled", body)
        self.assertIsNone(body.get("last_scan"))

    def test_news_channels_test_returns_unavailable_when_no_pipeline(self) -> None:
        st, body = _http_post(
            f"{self.base}/news/channels/test",
            token="secret-token-test",
            body={},
        )
        self.assertEqual(st, 200)
        assert body is not None
        self.assertFalse(body.get("available"))
        self.assertEqual(body["summary"]["probed"], 0)

    def test_alerts_pending_drains_queue(self) -> None:
        # Pause emits a BOT_PAUSED_RESUMED alert. Subscribe chat 101 first
        # so the alert fans out.
        _http_post(
            f"{self.base}/alerts/subscriptions",
            token="secret-token-test",
            body={"chat_id": 101, "type": "bot_paused_resumed", "enabled": True},
        )
        _http_post(f"{self.base}/pause", token="secret-token-test", body={"reason": "x"})

        st, body = _http_get(
            f"{self.base}/alerts/pending?chat_id=101",
            token="secret-token-test",
        )
        self.assertEqual(st, 200)
        assert body is not None
        types = [a["type"] for a in body["alerts"]]
        self.assertIn("bot_paused_resumed", types)

        # Second drain should be empty (alerts removed on read).
        _, body2 = _http_get(
            f"{self.base}/alerts/pending?chat_id=101",
            token="secret-token-test",
        )
        assert body2 is not None
        self.assertEqual(body2["alerts"], [])


if __name__ == "__main__":
    unittest.main()
