"""Tests for the cross-thread :class:`src.control.controller.BotController`.

We mock the eToro HTTP layer with a tiny stub that records calls and
returns canned portfolio payloads, plus an in-memory persistence
double, so the controller is exercised end-to-end without real
network or disk dependencies.
"""

import logging
import tempfile
import unittest
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
from src.config_store import ConfigStore
from src.control.controller import BotController, ControllerError
from src.persistence import StatePersistence
from src.state import BotState
from src.telemetry import TelemetryStore
from src.trade_history import TradeHistoryLog


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _StubEtoro:
    """Mimics the EtoroClient public surface used by the controller."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str]] = []

    def get(self, path: str, params=None, retries: int = 0):  # noqa: ARG002
        self.calls.append(("GET", path))
        return self.payload

    def post(self, path: str, json=None, retries: int = 0):  # noqa: ARG002
        self.calls.append(("POST", path))
        return {"orderForClose": {"orderID": 91234}}


def _make_app_config() -> AppConfig:
    return AppConfig(
        trading_mode="paper",
        guardrails=GuardrailsConfig(),
        operations=OperationsConfig(trade_spacing_seconds=0),  # tests run fast
        universe=UniverseConfig(),
        news=NewsConfig(enabled=False),
        fundamentals=FundamentalsConfig(enabled=False),
        strategy=StrategyConfig(),
        ai=AiConfig(enabled=False),
        tools=ToolsConfig(enabled=False),
        logging=LoggingConfig(),
        etoro=EtoroCredentials(
            public_key="x", user_key="y", is_real=False, allow_real=False,
        ),
        azure=AzureCredentials(endpoint=None, api_key=None, deployment=None),
        control=ControlServiceConfig(internal_api_token="t"),
        alerting=AlertingConfig(),
    )


def _make_controller(
    tmpdir: Path,
    etoro: _StubEtoro,
    *,
    config_store: ConfigStore | None = None,
    alerts=None,
) -> BotController:
    cfg = _make_app_config()
    state = BotState()
    log = logging.getLogger("test.controller")
    log.addHandler(logging.NullHandler())
    return BotController(
        cfg=cfg,
        state=state,
        etoro=etoro,
        ai_client=None,
        telemetry=TelemetryStore(),
        history=TradeHistoryLog(tmpdir / "history.jsonl"),
        persistence=StatePersistence(tmpdir / "state.json"),
        config_store=config_store,
        alerts=alerts,
        logger=log,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class PauseResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmpdir = Path(self._tmp.name)
        self.etoro = _StubEtoro({"clientPortfolio": {"credit": 10_000.0, "positions": []}})
        self.controller = _make_controller(self.tmpdir, self.etoro)

    def test_pause_then_resume(self) -> None:
        self.assertFalse(self.controller.paused)
        result = self.controller.pause(reason="test")
        self.assertTrue(self.controller.paused)
        self.assertTrue(result["paused"])
        self.assertFalse(result["was_already_paused"])

        result2 = self.controller.pause()
        self.assertTrue(result2["was_already_paused"])

        result3 = self.controller.resume()
        self.assertFalse(self.controller.paused)
        self.assertFalse(result3["was_already_running"])

    def test_pause_persists_to_disk(self) -> None:
        self.controller.pause()
        # Force load through a fresh persistence instance
        loaded, meta = StatePersistence(self.tmpdir / "state.json").load()
        self.assertIsNotNone(loaded)
        assert meta is not None
        self.assertTrue(meta.paused)

    def test_unhalt_clears_sticky_kill_switch(self) -> None:
        """``/unhalt`` is the operator's escape hatch when a previous
        drawdown set ``halted_today=true`` and the bot has done actions
        today (so the kill switch won't auto-clear). The method must
        flip the flag, wipe the baseline so the next cycle rebases, and
        report ``was_halted=true`` exactly once.
        """
        state = self.controller._state  # noqa: SLF001 — test scaffolding
        state.halted_today = True
        state.session_baseline_equity = 12_345.0
        result = self.controller.unhalt(reason="test")
        self.assertTrue(result["was_halted"])
        self.assertFalse(state.halted_today)
        self.assertIsNone(state.session_baseline_equity)

        # Second call is a no-op — there is nothing to clear.
        result2 = self.controller.unhalt()
        self.assertFalse(result2["was_halted"])


class GuardrailEditTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmpdir = Path(self._tmp.name)
        self.etoro = _StubEtoro({"clientPortfolio": {"credit": 0.0, "positions": []}})
        self.controller = _make_controller(self.tmpdir, self.etoro)

    def test_apply_change_propagates_to_shared_config(self) -> None:
        before = self.controller.get_guardrails()
        result = self.controller.apply_guardrails_change("max_per_trade_usd", "275")
        after = self.controller.get_guardrails()
        self.assertEqual(before["max_per_trade_usd"], 500.0)
        self.assertEqual(result["previous"], 500.0)
        self.assertEqual(result["current"], 275.0)
        self.assertEqual(after["max_per_trade_usd"], 275.0)

    def test_int_field_coerced(self) -> None:
        self.controller.apply_guardrails_change("max_parallel_trades", "4")
        self.assertEqual(self.controller.get_guardrails()["max_parallel_trades"], 4)

    def test_unknown_key_rejected(self) -> None:
        with self.assertRaises(ControllerError):
            self.controller.apply_guardrails_change("nonexistent", 1)

    def test_non_numeric_rejected(self) -> None:
        with self.assertRaises(ControllerError):
            self.controller.apply_guardrails_change("max_per_trade_usd", "lots")


class GuardrailPersistenceTests(unittest.TestCase):
    """Edits made through the controller must hit the SQLite override store."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmpdir = Path(self._tmp.name)
        self.etoro = _StubEtoro({"clientPortfolio": {"credit": 0.0, "positions": []}})
        self.store = ConfigStore(self.tmpdir / "config.sqlite")
        self.addCleanup(self.store.close)
        self.controller = _make_controller(
            self.tmpdir, self.etoro, config_store=self.store,
        )

    def test_apply_guardrails_change_writes_to_store(self) -> None:
        self.controller.apply_guardrails_change("default_stop_loss_pct", "7")
        persisted = self.store.get_section("guardrails")
        self.assertEqual(persisted["default_stop_loss_pct"], 7.0)

    def test_change_survives_store_close_and_reopen(self) -> None:
        """End-to-end: Telegram edit → DB → fresh process reads new value."""
        self.controller.apply_guardrails_change("default_take_profit_pct", "12")
        # Simulate restart by reopening the same DB file.
        self.store.close()
        fresh = ConfigStore(self.tmpdir / "config.sqlite")
        try:
            persisted = fresh.get_section("guardrails")
            self.assertEqual(persisted["default_take_profit_pct"], 12.0)
        finally:
            fresh.close()

    def test_controller_without_store_does_not_crash(self) -> None:
        controller = _make_controller(self.tmpdir, self.etoro, config_store=None)
        result = controller.apply_guardrails_change("default_stop_loss_pct", "9")
        # In-memory mutation still happens even when persistence is disabled.
        self.assertEqual(result["current"], 9.0)
        self.assertEqual(controller.get_guardrails()["default_stop_loss_pct"], 9.0)


class PanicCloseTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmpdir = Path(self._tmp.name)
        self.payload = {
            "clientPortfolio": {
                "credit": 10_000.0,
                "unrealizedPnL": 0.0,
                "positions": [
                    {
                        "positionID": 11, "instrumentID": 100, "isBuy": True,
                        "openRate": 100.0, "amount": 250.0, "units": 2.5,
                        "leverage": 1, "mirrorID": 0, "pnL": 5.0,
                    },
                    {
                        "positionID": 22, "instrumentID": 200, "isBuy": True,
                        "openRate": 50.0, "amount": 100.0, "units": 2.0,
                        "leverage": 1, "mirrorID": 0, "pnL": -3.0,
                    },
                    {  # mirror — should be skipped
                        "positionID": 33, "instrumentID": 300, "isBuy": True,
                        "openRate": 1.0, "amount": 10.0, "units": 10.0,
                        "leverage": 1, "mirrorID": 999, "pnL": 0.0,
                    },
                ],
                "ordersForOpen": [], "orders": [], "mirrors": [],
            }
        }
        self.etoro = _StubEtoro(self.payload)
        self.controller = _make_controller(self.tmpdir, self.etoro)

    def test_panic_all_closes_every_non_mirror_position(self) -> None:
        result = self.controller.panic_close_all(scope="all", reason="test")
        # 2 closes attempted (mirror skipped), both ok per stub.
        self.assertEqual(result["closed_attempted"], 2)
        self.assertEqual(result["closed_ok"], 2)
        self.assertTrue(self.controller.paused)
        # eToro POST was invoked twice with the close path.
        post_calls = [p for m, p in self.etoro.calls if m == "POST"]
        self.assertEqual(len(post_calls), 2)
        self.assertTrue(all("market-close-orders" in p for p in post_calls))

    def test_panic_bot_owned_only_closes_owned(self) -> None:
        # Mark only position 11 as bot-owned.
        self.controller._state.add_owned(11)  # noqa: SLF001 - test reaches in
        result = self.controller.panic_close_all(scope="bot_owned")
        self.assertEqual(result["closed_attempted"], 1)
        post_calls = [p for m, p in self.etoro.calls if m == "POST"]
        self.assertEqual(len(post_calls), 1)
        self.assertIn("/positions/11", post_calls[0])

    def test_panic_invalid_scope_raises(self) -> None:
        with self.assertRaises(ControllerError):
            self.controller.panic_close_all(scope="weird")

    def test_panic_records_history(self) -> None:
        self.controller.panic_close_all(scope="all")
        entries = self.controller.recent_history(limit=10)
        self.assertEqual(len(entries), 2)
        self.assertTrue(all(e["status"] == "panic_close" for e in entries))


class StatusSnapshotTests(unittest.TestCase):
    def test_status_contains_expected_fields(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        controller = _make_controller(
            Path(tmp.name),
            _StubEtoro({"clientPortfolio": {"credit": 0.0, "positions": []}}),
        )
        d = controller.snapshot_status_dict()
        for key in (
            "paused", "cycle_count", "trading_mode", "env_segment",
            "ai_enabled", "bot_owned_position_count",
        ):
            self.assertIn(key, d)
        self.assertEqual(d["trading_mode"], "paper")
        self.assertEqual(d["env_segment"], "demo")


class AlertsControlSurfaceTests(unittest.TestCase):
    """Verify the controller's /alerts management methods."""

    def setUp(self) -> None:
        from src.alerts import AlertHub, AlertSubscriptions, safety_only_default

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmpdir = Path(self._tmp.name)
        self.subs = AlertSubscriptions(
            self.tmpdir / "subs.json",
            default_set=safety_only_default(),
        )
        self.hub = AlertHub(
            allowed_chat_ids=(101, 202),
            subscriptions=self.subs,
        )
        self.controller = _make_controller(
            self.tmpdir,
            _StubEtoro({"clientPortfolio": {"credit": 0.0, "positions": []}}),
            alerts=self.hub,
        )

    def test_list_alert_types_includes_labels(self) -> None:
        types = self.controller.list_alert_types()
        type_names = {t["type"] for t in types}
        self.assertIn("trade_opened", type_names)
        self.assertIn("panic_close", type_names)
        labels = {t["label"] for t in types}
        self.assertTrue(any("Panic close" in label for label in labels))

    def test_get_subscriptions_seeds_default(self) -> None:
        result = self.controller.get_alert_subscriptions(101)
        self.assertEqual(result["chat_id"], 101)
        self.assertIn("panic_close", result["enabled"])
        self.assertIn("daily_loss_halt", result["enabled"])
        self.assertNotIn("trade_opened", result["enabled"])
        # Available list reflects current state for each type
        types_to_state = {a["type"]: a["enabled"] for a in result["available"]}
        self.assertTrue(types_to_state["panic_close"])
        self.assertFalse(types_to_state["trade_opened"])

    def test_set_subscription_enable(self) -> None:
        result = self.controller.set_alert_subscription(101, "trade_opened", True)
        self.assertEqual(result["type"], "trade_opened")
        self.assertTrue(result["enabled"])
        self.assertIn("trade_opened", result["all_enabled"])

    def test_toggle_subscription_flips(self) -> None:
        first = self.controller.toggle_alert_subscription(101, "panic_close")
        # Default state had panic_close ON; first toggle turns it OFF.
        self.assertFalse(first["enabled"])
        second = self.controller.toggle_alert_subscription(101, "panic_close")
        self.assertTrue(second["enabled"])

    def test_set_unknown_type_raises(self) -> None:
        with self.assertRaises(ControllerError):
            self.controller.set_alert_subscription(101, "ghost_alert", True)

    def test_pause_emits_to_subscribed_chats_only(self) -> None:
        # Chat 101 has BOT_PAUSED_RESUMED OFF by default; chat 202 enables it.
        self.controller.set_alert_subscription(202, "bot_paused_resumed", True)
        self.controller.pause(reason="test")
        self.assertEqual(self.controller.drain_alerts(101), [])
        chat_202 = self.controller.drain_alerts(202)
        self.assertEqual(len(chat_202), 1)
        self.assertEqual(chat_202[0]["type"], "bot_paused_resumed")

    def test_drain_returns_oldest_first_and_clears(self) -> None:
        from src.alerts import AlertType

        self.controller._alerts.emit(  # noqa: SLF001 - test reaches in
            AlertType.PANIC_CLOSE, title="A",
        )
        self.controller._alerts.emit(  # noqa: SLF001
            AlertType.PANIC_CLOSE, title="B",
        )
        first_batch = self.controller.drain_alerts(101)
        second_batch = self.controller.drain_alerts(101)
        self.assertEqual([a["title"] for a in first_batch], ["A", "B"])
        self.assertEqual(second_batch, [])

    def test_panic_close_emits_panic_alert(self) -> None:
        self.controller.panic_close_all(scope="bot_owned", reason="kill")
        # Both 101 and 202 default to PANIC_CLOSE on (safety_only).
        for chat in (101, 202):
            alerts = self.controller.drain_alerts(chat)
            self.assertEqual(len(alerts), 1)
            self.assertEqual(alerts[0]["type"], "panic_close")

    def test_no_alert_hub_disables_endpoints_gracefully(self) -> None:
        controller = _make_controller(
            self.tmpdir,
            _StubEtoro({"clientPortfolio": {"credit": 0.0, "positions": []}}),
            alerts=None,
        )
        with self.assertRaises(ControllerError):
            controller.get_alert_subscriptions(101)
        # drain returns empty list when hub is missing
        self.assertEqual(controller.drain_alerts(101), [])


if __name__ == "__main__":
    unittest.main()
