"""Tests for :meth:`BotController.snapshot_fundamentals`.

We re-use the controller-test scaffolding pattern but inline a tiny
fixture builder here so this module stays independent.
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
from src.control.controller import BotController
from src.fundamentals.cache import FundamentalsCache
from src.fundamentals.types import FundamentalsSnapshot
from src.persistence import StatePersistence
from src.state import BotState
from src.telemetry import TelemetryStore
from src.trade_history import TradeHistoryLog


class _StubEtoro:
    def get(self, *a, **kw):  # noqa: D401, ARG002
        return {}

    def post(self, *a, **kw):  # noqa: D401, ARG002
        return {}


class _StubFetcher:
    name = "stub"

    def fetch(self, symbol: str):
        return FundamentalsSnapshot(
            symbol=symbol.upper(),
            fetched_at_unix=1_000.0,
            name=f"{symbol.upper()} Inc.",
            sector="Tech",
            trailing_pe=20.0,
        )


def _make_app_config() -> AppConfig:
    return AppConfig(
        trading_mode="paper",
        guardrails=GuardrailsConfig(),
        operations=OperationsConfig(trade_spacing_seconds=0),
        universe=UniverseConfig(),
        news=NewsConfig(enabled=False),
        fundamentals=FundamentalsConfig(enabled=True, budget_per_refresh=10),
        strategy=StrategyConfig(),
        ai=AiConfig(enabled=False),
        tools=ToolsConfig(enabled=False),
        logging=LoggingConfig(),
        etoro=EtoroCredentials(public_key="x", user_key="y", is_real=False, allow_real=False),
        azure=AzureCredentials(endpoint=None, api_key=None, deployment=None),
        control=ControlServiceConfig(internal_api_token="t"),
        alerting=AlertingConfig(),
    )


class SnapshotFundamentalsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmpdir = Path(self._tmp.name)
        log = logging.getLogger("test.controller.fundamentals")
        log.addHandler(logging.NullHandler())
        self.controller = BotController(
            cfg=_make_app_config(),
            state=BotState(),
            etoro=_StubEtoro(),
            ai_client=None,
            telemetry=TelemetryStore(),
            history=TradeHistoryLog(tmpdir / "history.jsonl"),
            persistence=StatePersistence(tmpdir / "state.json"),
            logger=log,
        )
        self.cache = FundamentalsCache(
            fetcher=_StubFetcher(),
            path=tmpdir / "fundamentals.json",
            refresh_after_hours=24.0,
            clock=lambda: 1_000.0,
        )
        self.cache.refresh(["AAPL", "MSFT"])
        self.controller.set_fundamentals(self.cache)

    def test_list_view(self) -> None:
        out = self.controller.snapshot_fundamentals()
        self.assertTrue(out["enabled"])
        self.assertEqual(out["count"], 2)
        symbols = [it["symbol"] for it in out["items"]]
        self.assertEqual(sorted(symbols), ["AAPL", "MSFT"])

    def test_detail_view(self) -> None:
        out = self.controller.snapshot_fundamentals(symbol="aapl")
        self.assertTrue(out["enabled"])
        self.assertEqual(out["symbol"], "AAPL")
        self.assertIsNotNone(out["snapshot"])
        self.assertEqual(out["snapshot"]["name"], "AAPL Inc.")

    def test_unknown_symbol_returns_none_snapshot(self) -> None:
        out = self.controller.snapshot_fundamentals(symbol="NVDA")
        self.assertTrue(out["enabled"])
        self.assertEqual(out["symbol"], "NVDA")
        self.assertIsNone(out["snapshot"])

    def test_no_cache_wired_returns_disabled(self) -> None:
        # Build a separate controller without ``set_fundamentals`` call.
        log = logging.getLogger("test.controller.fundamentals.unwired")
        log.addHandler(logging.NullHandler())
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name)
        c = BotController(
            cfg=_make_app_config(),
            state=BotState(),
            etoro=_StubEtoro(),
            ai_client=None,
            telemetry=TelemetryStore(),
            history=TradeHistoryLog(path / "h.jsonl"),
            persistence=StatePersistence(path / "s.json"),
            logger=log,
        )
        out = c.snapshot_fundamentals()
        self.assertFalse(out["enabled"])
        self.assertEqual(out["items"], [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
