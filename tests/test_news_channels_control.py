"""Controller- and Telegram-formatter-level tests for `/channels`.

Covers:

* :meth:`BotController.snapshot_news_channels` — merges config intent
  + wired aggregator state + last_scan stats.
* :meth:`BotController.test_news_channels` — runs the live-probe path
  without folding results into the candidate store.
* :func:`format_channels_overview` / :func:`format_channels_test` —
  render the payloads as plain text suitable for Telegram.
* :func:`parse_channels_args` — argument parsing for `/channels [sub]
  [names]`.
"""

from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from typing import Iterable

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
from src.news.aggregator import NewsAggregator
from src.news.candidate_store import CandidateStore
from src.news.scheduler import NewsScheduler
from src.news.sources.base import NewsItem
from src.news.ticker_extractor import TickerExtractor
from src.persistence import StatePersistence
from src.state import BotState
from src.telegram_service.channel_formatter import (
    format_channels_logs,
    format_channels_overview,
    format_channels_test,
    parse_channels_args,
)
from src.telemetry import TelemetryStore
from src.trade_history import TradeHistoryLog


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _StubEtoro:
    def get(self, path, params=None, retries=0):  # noqa: ARG002
        return {}

    def post(self, path, json=None, retries=0):  # noqa: ARG002
        return {}


class _OkSource:
    name = "stocktwits"

    def __init__(self, items: list[NewsItem]) -> None:
        self._items = items

    def fetch(self, *, since=None, known_symbols=None) -> Iterable[NewsItem]:  # noqa: ARG002
        return list(self._items)


class _FailingSource:
    name = "google_news"

    def fetch(self, *, since=None, known_symbols=None):  # noqa: ARG002
        raise RuntimeError("boom")


class _DisabledSource:
    name = "sec_edgar"
    _disabled_reason = "SEC_USER_AGENT not configured"

    def fetch(self, *, since=None, known_symbols=None):  # noqa: ARG002
        raise AssertionError("fetch must not run on a disabled source")


def _item(symbol: str = "AAPL", headline: str = "Apple news") -> NewsItem:
    return NewsItem(
        source="stocktwits",
        symbols=(symbol,),
        headline=headline,
        url=f"https://example.test/{symbol}",
        published_at=1_700_000_000.0,
    )


def _app_cfg(*, enabled_sources: tuple[str, ...]) -> AppConfig:
    return AppConfig(
        trading_mode="paper",
        guardrails=GuardrailsConfig(),
        operations=OperationsConfig(trade_spacing_seconds=0),
        universe=UniverseConfig(),
        news=NewsConfig(enabled=True, enabled_sources=enabled_sources),
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


def _build_controller_with_pipeline(
    tmpdir: Path,
    sources: list,
    *,
    enabled_sources: tuple[str, ...] | None = None,
) -> tuple[BotController, NewsScheduler]:
    if enabled_sources is None:
        enabled_sources = tuple(getattr(s, "name", "?") for s in sources)
    store = CandidateStore(path=tmpdir / "cands.json", ttl_seconds=24 * 3600)
    aggregator = NewsAggregator(
        sources=sources,
        store=store,
        ticker_extractor=TickerExtractor(known_symbols=["AAPL", "MSFT"]),
    )
    scheduler = NewsScheduler(aggregator, interval_minutes=60)
    log = logging.getLogger("test.news.channels")
    log.addHandler(logging.NullHandler())
    controller = BotController(
        cfg=_app_cfg(enabled_sources=enabled_sources),
        state=BotState(),
        etoro=_StubEtoro(),
        ai_client=None,
        telemetry=TelemetryStore(),
        history=TradeHistoryLog(tmpdir / "history.jsonl"),
        persistence=StatePersistence(tmpdir / "state.json"),
        alerts=None,
        logger=log,
    )
    controller.set_news_store(store, scheduler)
    return controller, scheduler


# ---------------------------------------------------------------------------
# snapshot_news_channels
# ---------------------------------------------------------------------------

class SnapshotNewsChannelsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmpdir = Path(self._tmp.name)

    def test_overview_merges_config_intent_and_wired_state(self) -> None:
        controller, _ = _build_controller_with_pipeline(
            self.tmpdir,
            [_OkSource([]), _DisabledSource()],
            # config lists an extra unknown source — should still show up
            # with wired=False so the operator notices the typo.
            enabled_sources=("stocktwits", "sec_edgar", "yfinance"),
        )
        payload = controller.snapshot_news_channels()
        names = [c["name"] for c in payload["channels"]]
        self.assertEqual(set(names), {"stocktwits", "sec_edgar", "yfinance"})
        by_name = {c["name"]: c for c in payload["channels"]}
        self.assertTrue(by_name["stocktwits"]["wired"])
        self.assertTrue(by_name["stocktwits"]["enabled"])
        self.assertTrue(by_name["sec_edgar"]["wired"])
        self.assertEqual(
            by_name["sec_edgar"]["disabled_reason"],
            "SEC_USER_AGENT not configured",
        )
        self.assertFalse(by_name["yfinance"]["wired"])
        self.assertTrue(by_name["yfinance"]["enabled"])
        self.assertTrue(payload["pipeline_enabled"])
        self.assertEqual(payload["scan_interval_minutes"], 60)

    def test_last_scan_counts_are_attached(self) -> None:
        ok = _OkSource([_item("AAPL")])
        failing = _FailingSource()
        controller, scheduler = _build_controller_with_pipeline(
            self.tmpdir, [ok, failing],
        )
        scheduler.maybe_run(force=True)
        payload = controller.snapshot_news_channels()
        by_name = {c["name"]: c for c in payload["channels"]}
        self.assertEqual(by_name["stocktwits"]["last_items_kept"], 1)
        self.assertIsNone(by_name["stocktwits"]["last_error"])
        self.assertEqual(by_name["google_news"]["last_items_kept"], 0)
        self.assertIn("boom", by_name["google_news"]["last_error"] or "")
        self.assertEqual(payload["last_scan"]["items_kept"], 1)

    def test_returns_empty_when_pipeline_not_wired(self) -> None:
        # Build a controller without calling set_news_store
        log = logging.getLogger("test.news.channels.empty")
        log.addHandler(logging.NullHandler())
        controller = BotController(
            cfg=_app_cfg(enabled_sources=("stocktwits", "sec_edgar")),
            state=BotState(),
            etoro=_StubEtoro(),
            ai_client=None,
            telemetry=TelemetryStore(),
            history=TradeHistoryLog(self.tmpdir / "history.jsonl"),
            persistence=StatePersistence(self.tmpdir / "state.json"),
            logger=log,
        )
        payload = controller.snapshot_news_channels()
        self.assertEqual(
            {c["name"] for c in payload["channels"]},
            {"stocktwits", "sec_edgar"},
        )
        self.assertTrue(all(not c["wired"] for c in payload["channels"]))
        self.assertIsNone(payload["last_scan"])


# ---------------------------------------------------------------------------
# test_news_channels (live dry-run path)
# ---------------------------------------------------------------------------

class TestNewsChannelsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmpdir = Path(self._tmp.name)

    def test_dry_run_does_not_mutate_candidate_store(self) -> None:
        ok = _OkSource([_item("AAPL"), _item("MSFT")])
        controller, _ = _build_controller_with_pipeline(self.tmpdir, [ok])
        # store starts empty
        self.assertEqual(controller.snapshot_news()["candidates"], [])
        out = controller.test_news_channels()
        self.assertTrue(out["available"])
        self.assertEqual(out["summary"]["probed"], 1)
        self.assertEqual(out["summary"]["ok"], 1)
        # store stays empty — probe must not fold into the store
        self.assertEqual(controller.snapshot_news()["candidates"], [])

    def test_only_filter_runs_subset(self) -> None:
        ok = _OkSource([_item("AAPL")])
        failing = _FailingSource()
        controller, _ = _build_controller_with_pipeline(
            self.tmpdir, [ok, failing],
        )
        out = controller.test_news_channels(only=["stocktwits"])
        self.assertEqual(out["summary"]["probed"], 1)
        self.assertEqual(out["results"][0]["name"], "stocktwits")

    def test_disabled_source_reports_reason_without_calling_fetch(self) -> None:
        controller, _ = _build_controller_with_pipeline(
            self.tmpdir, [_DisabledSource()],
        )
        out = controller.test_news_channels()
        self.assertEqual(out["summary"]["probed"], 1)
        self.assertEqual(out["summary"]["ok"], 0)
        result = out["results"][0]
        self.assertFalse(result["ok"])
        self.assertEqual(
            result["disabled_reason"], "SEC_USER_AGENT not configured",
        )

    def test_no_pipeline_returns_available_false(self) -> None:
        log = logging.getLogger("test.news.channels.test")
        log.addHandler(logging.NullHandler())
        controller = BotController(
            cfg=_app_cfg(enabled_sources=("stocktwits",)),
            state=BotState(),
            etoro=_StubEtoro(),
            ai_client=None,
            telemetry=TelemetryStore(),
            history=TradeHistoryLog(self.tmpdir / "history.jsonl"),
            persistence=StatePersistence(self.tmpdir / "state.json"),
            logger=log,
        )
        out = controller.test_news_channels()
        self.assertFalse(out["available"])
        self.assertEqual(out["summary"]["probed"], 0)


# ---------------------------------------------------------------------------
# Telegram-side renderers + argument parser
# ---------------------------------------------------------------------------

class ChannelsFormatterTests(unittest.TestCase):
    def test_parse_overview(self) -> None:
        self.assertEqual(parse_channels_args(""), ("", []))

    def test_parse_test_with_names(self) -> None:
        sub, names = parse_channels_args("test stocktwits, yfinance google_news")
        self.assertEqual(sub, "test")
        self.assertEqual(names, ["stocktwits", "yfinance", "google_news"])

    def test_parse_logs(self) -> None:
        sub, names = parse_channels_args("logs")
        self.assertEqual((sub, names), ("logs", []))

    def test_bare_name_list_is_treated_as_test(self) -> None:
        sub, names = parse_channels_args("stocktwits,yfinance")
        self.assertEqual(sub, "test")
        self.assertEqual(names, ["stocktwits", "yfinance"])

    def test_overview_renders_summary_and_table(self) -> None:
        payload = {
            "pipeline_enabled": True,
            "scan_interval_minutes": 60,
            "ttl_hours": 24,
            "half_life_hours": 6.0,
            "channels": [
                {
                    "name": "stocktwits", "enabled": True, "wired": True,
                    "class": "X", "disabled_reason": None, "weight": 1.0,
                    "last_items_kept": 12, "last_error": None,
                },
                {
                    "name": "sec_edgar", "enabled": True, "wired": True,
                    "class": "Y",
                    "disabled_reason": "SEC_USER_AGENT not set",
                    "weight": 1.2, "last_items_kept": 0, "last_error": None,
                },
            ],
            "last_scan": {
                "finished_at_unix": 1_700_000_500.0,
                "items_kept": 12, "observations_recorded": 18,
            },
            "next_scan_in_seconds": 60 * 30,
        }
        text = format_channels_overview(payload)
        self.assertIn("[CHANNELS]", text)
        self.assertIn("pipeline:  on", text)
        self.assertIn("stocktwits", text)
        self.assertIn("DISABLED", text)
        self.assertIn("Issues:", text)
        self.assertIn("SEC_USER_AGENT not set", text)

    def test_test_payload_table_and_failures(self) -> None:
        payload = {
            "available": True,
            "summary": {"probed": 2, "ok": 1, "failed": 1},
            "results": [
                {
                    "name": "stocktwits", "ok": True, "items_count": 3,
                    "sample_headline": "Apple ships chips", "duration_ms": 120,
                    "error": None, "disabled_reason": None,
                },
                {
                    "name": "google_news", "ok": False, "items_count": 0,
                    "sample_headline": None, "duration_ms": 80,
                    "error": "RuntimeError: 503", "disabled_reason": None,
                },
            ],
        }
        text = format_channels_test(payload)
        self.assertIn("probed=2", text)
        self.assertIn("ok=1", text)
        self.assertIn("failed=1", text)
        self.assertIn("stocktwits", text)
        self.assertIn("Apple ships chips", text)
        self.assertIn("Failures:", text)
        self.assertIn("google_news", text)

    def test_test_payload_unavailable(self) -> None:
        text = format_channels_test({"available": False})
        self.assertIn("isn't wired", text)

    def test_logs_payload_renders_per_source_errors(self) -> None:
        payload = {
            "scan_interval_minutes": 60,
            "channels": [
                {
                    "name": "sec_edgar",
                    "disabled_reason": "missing UA",
                },
            ],
            "last_scan": {
                "started_at_unix": 1_700_000_000.0,
                "finished_at_unix": 1_700_000_300.0,
                "items_fetched": 42, "items_kept": 30,
                "observations_recorded": 55,
                "per_source_counts": {"stocktwits": 12, "google_news": 0},
                "per_source_errors": {"google_news": "RuntimeError: timeout"},
            },
        }
        text = format_channels_logs(payload)
        self.assertIn("most recent scan", text)
        self.assertIn("stocktwits", text)
        self.assertIn("google_news", text)
        self.assertIn("timeout", text)
        self.assertIn("Configured-but-disabled", text)


if __name__ == "__main__":
    unittest.main()
