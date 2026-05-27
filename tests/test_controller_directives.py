"""Tests for the directives + token-usage controller surface.

Exercises:

* ``snapshot_directives`` before / after the store is wired.
* ``set_directive`` / ``clear_directive`` / ``set_directive_note``
  round-trips, with persistence to disk and a fresh-load restore.
* ``snapshot_token_usage`` before / after the tracker is wired.
"""

from __future__ import annotations

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
from src.control.controller import BotController, ControllerError
from src.persistence import StatePersistence
from src.state import BotState
from src.strategy.directives import Directives, DirectivesStore
from src.telemetry import TelemetryStore
from src.trade_history import TradeHistoryLog


class _StubEtoro:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get(self, path: str, params=None, retries: int = 0):  # noqa: ARG002
        self.calls.append(path)
        return {"clientPortfolio": {"credit": 0.0, "positions": []}}

    def post(self, path: str, json=None, retries: int = 0):  # noqa: ARG002
        return {}


class _StubTokenTracker:
    def __init__(self) -> None:
        self.snapshot_calls = 0

    def snapshot(self) -> dict:
        self.snapshot_calls += 1
        return {
            "deployment": "gpt-5-mini",
            "rates": {
                "family": "gpt-5-mini",
                "input_per_m": 0.25,
                "cached_per_m": 0.03,
                "output_per_m": 2.00,
            },
            "today": {
                "calls": 3, "prompt_tokens": 100, "cached_tokens": 0,
                "completion_tokens": 50, "cost_usd": 0.01,
            },
            "all_time": {
                "calls": 3, "prompt_tokens": 100, "cached_tokens": 0,
                "completion_tokens": 50, "cost_usd": 0.01,
            },
            "by_call_type": {},
            "last_call": None,
            "recent_count": 3,
        }


def _app_cfg() -> AppConfig:
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
        control=ControlServiceConfig(internal_api_token="t"),
        alerting=AlertingConfig(),
    )


def _make_controller(tmpdir: Path) -> tuple[BotController, StatePersistence]:
    persistence = StatePersistence(tmpdir / "state.json")
    log = logging.getLogger("test.controller.dx")
    log.addHandler(logging.NullHandler())
    controller = BotController(
        cfg=_app_cfg(),
        state=BotState(),
        etoro=_StubEtoro(),
        ai_client=None,
        telemetry=TelemetryStore(),
        history=TradeHistoryLog(tmpdir / "history.jsonl"),
        persistence=persistence,
        config_store=None,
        alerts=None,
        logger=log,
    )
    return controller, persistence


class DirectivesControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmpdir = Path(self._tmp.name)
        self.controller, self.persistence = _make_controller(self.tmpdir)

    def test_snapshot_before_wire_is_disabled(self) -> None:
        snap = self.controller.snapshot_directives()
        self.assertEqual(snap["enabled"], False)
        self.assertIn("structured_keys", snap)
        # The default values block must still be present so the
        # Telegram renderer doesn't crash before the store is wired.
        self.assertIn("values", snap)

    def test_set_directive_without_store_raises(self) -> None:
        with self.assertRaises(ControllerError):
            self.controller.set_directive("no_overnight", "true")
        with self.assertRaises(ControllerError):
            self.controller.set_directive_note("hello")

    def test_set_directive_persists_to_disk(self) -> None:
        store = DirectivesStore()
        self.controller.set_directives_store(store)
        result = self.controller.set_directive("no_overnight", "true")
        self.assertFalse(result["previous"])
        self.assertTrue(result["current"])

        snap = self.controller.snapshot_directives()
        self.assertTrue(snap["enabled"])
        self.assertTrue(snap["values"]["no_overnight"])

        # Fresh load through persistence layer must surface the directive.
        _state, meta = self.persistence.load()
        loaded = self.persistence.load_directives()
        assert loaded is not None
        self.assertTrue(loaded["no_overnight"])

    def test_set_blocked_symbols_round_trip(self) -> None:
        self.controller.set_directives_store(DirectivesStore())
        self.controller.set_directive("blocked_symbols", "NVDA, TSLA")
        snap = self.controller.snapshot_directives()
        self.assertEqual(
            list(snap["values"]["blocked_symbols"]), ["NVDA", "TSLA"],
        )

    def test_clear_directive_resets_to_default(self) -> None:
        self.controller.set_directives_store(DirectivesStore())
        self.controller.set_directive("hold_ceiling_minutes", "120")
        cleared = self.controller.clear_directive("hold_ceiling_minutes")
        self.assertEqual(cleared["previous"], 120)
        self.assertEqual(cleared["current"], 0)

    def test_invalid_directive_value_raises(self) -> None:
        self.controller.set_directives_store(DirectivesStore())
        with self.assertRaises(ControllerError):
            self.controller.set_directive("hold_ceiling_minutes", "negative")
        with self.assertRaises(ControllerError):
            self.controller.set_directive("does_not_exist", "x")

    def test_note_set_and_clear(self) -> None:
        self.controller.set_directives_store(DirectivesStore())
        r1 = self.controller.set_directive_note("watch financials")
        self.assertEqual(r1["previous"], "")
        self.assertEqual(r1["current"], "watch financials")
        r2 = self.controller.clear_directive_note()
        self.assertEqual(r2["previous"], "watch financials")
        self.assertEqual(r2["current"], "")


class TokenUsageControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmpdir = Path(self._tmp.name)
        self.controller, _ = _make_controller(self.tmpdir)

    def test_snapshot_before_wire_is_disabled(self) -> None:
        snap = self.controller.snapshot_token_usage()
        self.assertEqual(snap, {"enabled": False})

    def test_snapshot_after_wire_returns_payload(self) -> None:
        tracker = _StubTokenTracker()
        self.controller.set_token_usage_tracker(tracker)
        snap = self.controller.snapshot_token_usage()
        self.assertTrue(snap["enabled"])
        self.assertEqual(snap["deployment"], "gpt-5-mini")
        self.assertEqual(snap["today"]["calls"], 3)
        self.assertEqual(tracker.snapshot_calls, 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
