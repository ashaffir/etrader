"""Tests for src/strategy/earnings_calendar.py."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.strategy.earnings_calendar import (
    EarningsCalendarCache,
    EarningsEntry,
)


def _entry(
    symbol: str,
    *,
    hours_from_now: float,
    now: datetime | None = None,
) -> EarningsEntry:
    """Build an :class:`EarningsEntry` ``hours_from_now`` from ``now``.

    ``now`` is captured up-front so the test's read-side (``hours_until``)
    can pass the same instant in and avoid scheduling jitter near
    integer day boundaries.
    """
    anchor = now or datetime.now(timezone.utc)
    when = anchor + timedelta(hours=hours_from_now)
    return EarningsEntry(
        symbol=symbol.upper(),
        earnings_at_utc=when,
        fetched_at_unix=anchor.timestamp(),
    )


class EarningsEntryTests(unittest.TestCase):
    def test_hours_and_days_until_match(self) -> None:
        now = datetime(2026, 5, 27, 14, 0, 0, tzinfo=timezone.utc)
        e = _entry("AAPL", hours_from_now=49.0, now=now)
        self.assertAlmostEqual(e.hours_until(now=now), 49.0, places=5)
        self.assertEqual(e.days_until(now=now), 2)

    def test_negative_for_past_event(self) -> None:
        past = datetime.now(timezone.utc) - timedelta(hours=4)
        e = EarningsEntry("AAPL", past, fetched_at_unix=0.0)
        self.assertLess(e.hours_until(), 0)

    def test_roundtrip_via_dict(self) -> None:
        e = _entry("AAPL", hours_from_now=24.0)
        loaded = EarningsEntry.from_dict(e.to_dict())
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.symbol, "AAPL")
        self.assertAlmostEqual(
            loaded.earnings_at_utc.timestamp(),
            e.earnings_at_utc.timestamp(),
            places=3,
        )

    def test_from_dict_returns_none_on_garbage(self) -> None:
        self.assertIsNone(EarningsEntry.from_dict({}))
        self.assertIsNone(EarningsEntry.from_dict({"symbol": "X"}))
        self.assertIsNone(
            EarningsEntry.from_dict(
                {"symbol": "X", "earnings_at_utc": "not-a-date"}
            )
        )


class _FakeFetcher:
    """Deterministic fetcher used to avoid yfinance in tests."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.responses: dict[str, EarningsEntry | None] = {}

    def __call__(self, symbol: str) -> EarningsEntry | None:
        self.calls.append(symbol)
        return self.responses.get(symbol)


class EarningsCalendarCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "cache.json"
        self.fetcher = _FakeFetcher()
        self.cache = EarningsCalendarCache(
            self.path, ttl_seconds=3600, fetcher=self.fetcher,
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_refresh_persists_to_disk(self) -> None:
        self.fetcher.responses["AAPL"] = _entry("AAPL", hours_from_now=24.0)
        entry = self.cache.refresh("AAPL")
        self.assertIsNotNone(entry)
        # File exists and contains the entry.
        raw = json.loads(self.path.read_text())
        self.assertIn("AAPL", raw["entries"])

    def test_get_returns_fresh_entry(self) -> None:
        self.fetcher.responses["AAPL"] = _entry("AAPL", hours_from_now=24.0)
        self.cache.refresh("AAPL")
        got = self.cache.get("AAPL")
        self.assertIsNotNone(got)
        assert got is not None
        self.assertEqual(got.symbol, "AAPL")

    def test_get_drops_past_event(self) -> None:
        self.fetcher.responses["AAPL"] = EarningsEntry(
            symbol="AAPL",
            earnings_at_utc=datetime.now(timezone.utc) - timedelta(hours=1),
            fetched_at_unix=datetime.now(timezone.utc).timestamp(),
        )
        self.cache.refresh("AAPL")
        self.assertIsNone(self.cache.get("AAPL"))

    def test_refresh_skips_when_within_ttl(self) -> None:
        self.fetcher.responses["AAPL"] = _entry("AAPL", hours_from_now=24.0)
        self.cache.refresh("AAPL")
        self.cache.refresh("AAPL")  # within TTL — should NOT call fetcher again
        self.assertEqual(self.fetcher.calls, ["AAPL"])

    def test_force_refresh_overrides_ttl(self) -> None:
        self.fetcher.responses["AAPL"] = _entry("AAPL", hours_from_now=24.0)
        self.cache.refresh("AAPL")
        self.cache.refresh("AAPL", force=True)
        self.assertEqual(self.fetcher.calls, ["AAPL", "AAPL"])

    def test_negative_result_is_cached(self) -> None:
        # No response queued → fetcher returns None.
        self.cache.refresh("UNKNOWN")
        self.cache.refresh("UNKNOWN")
        # Second call is within negative-TTL window (half the TTL).
        self.assertEqual(self.fetcher.calls, ["UNKNOWN"])

    def test_persisted_state_loads_on_reopen(self) -> None:
        self.fetcher.responses["AAPL"] = _entry("AAPL", hours_from_now=24.0)
        self.cache.refresh("AAPL")
        # Reopen cache → should hydrate from disk, no fetch needed.
        fresh_fetcher = _FakeFetcher()
        reloaded = EarningsCalendarCache(
            self.path, ttl_seconds=3600, fetcher=fresh_fetcher,
        )
        self.assertIsNotNone(reloaded.get("AAPL"))
        self.assertEqual(fresh_fetcher.calls, [])

    def test_fetcher_exception_does_not_propagate(self) -> None:
        def boom(_symbol: str) -> EarningsEntry | None:
            raise RuntimeError("yahoo is down")
        cache = EarningsCalendarCache(
            self.path, ttl_seconds=3600, fetcher=boom,
        )
        self.assertIsNone(cache.refresh("AAPL"))


class _StaticLookup:
    """Behave like ``EarningsCalendarCache.get`` for the proximity tool."""

    def __init__(self, entry: EarningsEntry | None = None, *, raises: bool = False) -> None:
        self._entry = entry
        self._raises = raises

    def __call__(self, _symbol: str) -> Any:
        if self._raises:
            raise RuntimeError("kaboom")
        return self._entry


class EarningsProximityToolTests(unittest.TestCase):
    def _make_ctx(
        self,
        *,
        candidate_action: str = "BUY",
        lookup: Any = None,
        blackout_hours: int = 0,
    ):
        from src.config import GuardrailsConfig, StrategyConfig
        from src.strategy.tools.base import AssetClass, ToolContext

        # Minimal ctx; we don't need candles for this tool.
        return ToolContext(
            instrument_id=1,
            symbol="AAPL",
            asset_class=AssetClass.STOCK,
            candidate_action=candidate_action,
            strategy=StrategyConfig(),
            guardrails=GuardrailsConfig(
                pre_earnings_buy_blackout_hours=blackout_hours
            ),
            candles=[],
            rate=None,
            instrument_meta=None,
            earnings_lookup=lookup,
        )

    def test_no_lookup_is_noop(self) -> None:
        from src.strategy.tools.context_tools import EarningsProximityTool
        ctx = self._make_ctx(lookup=None)
        result = EarningsProximityTool().evaluate(ctx)
        self.assertTrue(result.gate_passed)
        self.assertIsNone(result.features["hours_to_earnings"])

    def test_emits_features_when_entry_present(self) -> None:
        from src.strategy.tools.context_tools import EarningsProximityTool
        # 49h dodges the 48h integer-day boundary where sub-second
        # scheduling jitter would otherwise flip days_to_earnings.
        lookup = _StaticLookup(_entry("AAPL", hours_from_now=49.0))
        result = EarningsProximityTool().evaluate(self._make_ctx(lookup=lookup))
        self.assertIsNotNone(result.features["hours_to_earnings"])
        self.assertEqual(result.features["days_to_earnings"], 2)
        self.assertTrue(result.gate_passed)

    def test_gate_vetoes_buy_inside_blackout(self) -> None:
        from src.strategy.tools.context_tools import EarningsProximityTool
        lookup = _StaticLookup(_entry("AAPL", hours_from_now=12.0))
        result = EarningsProximityTool().evaluate(
            self._make_ctx(lookup=lookup, blackout_hours=24)
        )
        self.assertFalse(result.gate_passed)
        self.assertIn("blackout", result.gate_reason)

    def test_gate_silent_when_blackout_zero(self) -> None:
        from src.strategy.tools.context_tools import EarningsProximityTool
        lookup = _StaticLookup(_entry("AAPL", hours_from_now=12.0))
        result = EarningsProximityTool().evaluate(
            self._make_ctx(lookup=lookup, blackout_hours=0)
        )
        self.assertTrue(result.gate_passed)

    def test_gate_does_not_fire_on_close_action(self) -> None:
        from src.strategy.tools.context_tools import EarningsProximityTool
        lookup = _StaticLookup(_entry("AAPL", hours_from_now=12.0))
        result = EarningsProximityTool().evaluate(
            self._make_ctx(
                candidate_action="CLOSE", lookup=lookup, blackout_hours=24,
            )
        )
        self.assertTrue(result.gate_passed)

    def test_lookup_exception_treated_as_no_data(self) -> None:
        from src.strategy.tools.context_tools import EarningsProximityTool
        result = EarningsProximityTool().evaluate(
            self._make_ctx(lookup=_StaticLookup(raises=True), blackout_hours=24)
        )
        # Tool must never break a cycle — gate stays open and the
        # feature is None so the LLM knows we have no data.
        self.assertTrue(result.gate_passed)
        self.assertIsNone(result.features["hours_to_earnings"])


if __name__ == "__main__":
    unittest.main()
