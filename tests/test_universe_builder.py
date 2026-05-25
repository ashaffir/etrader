"""Tests for the phase-2 UniverseBuilder (news-driven + activity filter)."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config import OperationsConfig, UniverseConfig
from src.etoro.instrument_cache import InstrumentCache
from src.etoro.market_data import Candle, LiveRate
from src.news.candidate_store import CandidateStore
from src.strategy.activity_filter import ActivityDecision, ActivityFilter
from src.strategy.universe import UniverseBuilder


class _StubFilter(ActivityFilter):
    """Activity filter with a hard-coded verdict per symbol.

    ``decisions``: ``{instrument_id: ActivityDecision}``.
    Missing instrument IDs auto-pass (so tests can omit boilerplate).
    """

    def __init__(self, decisions: dict[int, ActivityDecision]) -> None:
        # Build a real underlying filter with permissive thresholds so
        # any super() calls don't surprise. Then override evaluate().
        super().__init__(UniverseConfig(min_atr_pct=0.0, max_spread_pct=999.0))
        self._scripted = dict(decisions)

    def evaluate(self, *, candles, rate) -> ActivityDecision:  # type: ignore[override]
        # Resolve by checking the symbol/id we baked into candles.
        # Tests inject `instrument_id=N` into candles to identify it.
        inst_id = candles[0].instrument_id if candles else 0
        if inst_id in self._scripted:
            return self._scripted[inst_id]
        return ActivityDecision(passed=True, reason="ok (auto)", atr_pct=1.0, spread_pct=0.1)


class _StubEtoro:
    """Fake EtoroClient. Records calls for assertions if needed."""


def _candles(inst_id: int, n: int = 30) -> list[Candle]:
    """Generate enough candles to look "real" to the universe builder."""
    return [
        Candle(
            instrument_id=inst_id,
            from_date=None,
            open=100.0 + i * 0.1,
            high=100.5 + i * 0.1,
            low=99.5 + i * 0.1,
            close=100.0 + i * 0.1,
            volume=1000.0,
        )
        for i in range(n)
    ]


def _rate(inst_id: int) -> LiveRate:
    return LiveRate(instrument_id=inst_id, ask=101.0, bid=100.0, last=100.5, timestamp=None)


def _make_builder(
    cfg: UniverseConfig,
    *,
    cache: InstrumentCache,
    store: CandidateStore,
    decisions: dict[int, ActivityDecision] | None = None,
    ai_client=None,
):
    return UniverseBuilder(
        cfg,
        OperationsConfig(),
        cache=cache,
        candidate_store=store,
        activity_filter=_StubFilter(decisions or {}),
        ai_client=ai_client,
        etoro_client=_StubEtoro(),
    )


def _resolve_table(table: dict[str, int]):
    """Build a patcher tuple for fetch_rates/fetch_candles/search_instrument."""
    def fake_search(_client, sym):
        return table.get(sym.upper())

    def fake_rates(_client, ids):
        return {i: _rate(i) for i in ids}

    def fake_candles(_client, inst_id, **_kwargs):
        return _candles(inst_id)

    return fake_search, fake_rates, fake_candles


class UniverseBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.cache = InstrumentCache.load(self.tmp / "cache.json")
        self.store = CandidateStore(path=self.tmp / "cands.json", ttl_seconds=24 * 3600)

    def _record(self, sym: str, *, score: float = 1.0, source: str = "stocktwits") -> None:
        self.store.record(symbol=sym, source=source, headline=f"{sym} mover", weight=score)

    def test_news_candidates_admitted_through_filter(self) -> None:
        self._record("AAPL", score=2.0)
        self._record("MSFT", score=1.0)
        table = {"AAPL": 1001, "MSFT": 1002}

        cfg = UniverseConfig(max_tracked=5, enable_llm_rotation=False)
        builder = _make_builder(cfg, cache=self.cache, store=self.store)
        s, r, c = _resolve_table(table)
        with patch("src.strategy.universe.search_instrument", side_effect=s), \
             patch("src.strategy.universe.fetch_rates", side_effect=r), \
             patch("src.strategy.universe.fetch_candles", side_effect=c):
            uni = builder.build()
        self.assertEqual(sorted(uni.symbol_for_id.values()), ["AAPL", "MSFT"])
        self.assertEqual(uni.source_counts.get("news"), 2)
        for sym in ("AAPL", "MSFT"):
            inst_id = table[sym]
            self.assertIn("atr=", uni.reason_for_id[inst_id])

    def test_rejections_recorded_when_filter_fails(self) -> None:
        self._record("AAPL", score=2.0)
        self._record("MSFT", score=1.0)
        table = {"AAPL": 1001, "MSFT": 1002}

        cfg = UniverseConfig(max_tracked=5, enable_llm_rotation=False)
        decisions = {
            1002: ActivityDecision(passed=False, reason="flat", atr_pct=0.05, spread_pct=0.1),
        }
        builder = _make_builder(cfg, cache=self.cache, store=self.store, decisions=decisions)
        s, r, c = _resolve_table(table)
        with patch("src.strategy.universe.search_instrument", side_effect=s), \
             patch("src.strategy.universe.fetch_rates", side_effect=r), \
             patch("src.strategy.universe.fetch_candles", side_effect=c):
            uni = builder.build()
        self.assertEqual(list(uni.symbol_for_id.values()), ["AAPL"])
        self.assertEqual(uni.rejected, {"MSFT": "flat"})

    def test_universe_shrinks_when_no_candidates(self) -> None:
        # No news, no LLM client, no seed → universe should be empty.
        cfg = UniverseConfig(max_tracked=5, enable_llm_rotation=False)
        builder = _make_builder(cfg, cache=self.cache, store=self.store)
        with patch("src.strategy.universe.search_instrument", return_value=None), \
             patch("src.strategy.universe.fetch_rates", return_value={}), \
             patch("src.strategy.universe.fetch_candles", return_value=[]):
            uni = builder.build()
        self.assertEqual(len(uni), 0)
        self.assertEqual(uni.source_counts, {})

    def test_must_include_bypasses_filter_and_cap(self) -> None:
        # Cap = 1, must_include AAPL (owned), one news candidate MSFT.
        # AAPL skips filter, MSFT goes through filter; both make it.
        self._record("MSFT", score=1.0)
        table = {"MSFT": 1002}

        cfg = UniverseConfig(max_tracked=1, enable_llm_rotation=False)
        builder = _make_builder(cfg, cache=self.cache, store=self.store)
        s, r, c = _resolve_table(table)
        with patch("src.strategy.universe.search_instrument", side_effect=s), \
             patch("src.strategy.universe.fetch_rates", side_effect=r), \
             patch("src.strategy.universe.fetch_candles", side_effect=c):
            uni = builder.build(must_include={1001: "AAPL"})

        # AAPL is owned → included. MSFT is news → cap was 1, but the
        # cap measures *discretionary* slots, and after subtracting the
        # 1 owned slot, no discretionary slots remain → MSFT is deferred.
        self.assertIn(1001, uni.instrument_ids)
        self.assertEqual(uni.source_counts.get("owned"), 1)
        self.assertEqual(uni.reason_for_id[1001], "owned position (auto-included)")
        # MSFT shows up in rejected with a deferral note.
        self.assertIn("MSFT", uni.rejected)

    def test_unresolvable_symbol_silently_skipped(self) -> None:
        self._record("AAPL", score=2.0)
        self._record("UNKNOWN", score=1.5)
        table = {"AAPL": 1001}  # UNKNOWN won't resolve

        cfg = UniverseConfig(max_tracked=5, enable_llm_rotation=False)
        builder = _make_builder(cfg, cache=self.cache, store=self.store)
        s, r, c = _resolve_table(table)
        with patch("src.strategy.universe.search_instrument", side_effect=s), \
             patch("src.strategy.universe.fetch_rates", side_effect=r), \
             patch("src.strategy.universe.fetch_candles", side_effect=c):
            uni = builder.build()
        self.assertEqual(list(uni.symbol_for_id.values()), ["AAPL"])
        self.assertNotIn("UNKNOWN", uni.rejected)  # never made it that far

    def test_seed_symbols_pass_through_filter(self) -> None:
        # No news candidates. Seed list with AAPL. Filter passes it.
        table = {"AAPL": 1001}
        cfg = UniverseConfig(
            base_symbols=("AAPL",),
            max_tracked=5,
            enable_llm_rotation=False,
        )
        builder = _make_builder(cfg, cache=self.cache, store=self.store)
        s, r, c = _resolve_table(table)
        with patch("src.strategy.universe.search_instrument", side_effect=s), \
             patch("src.strategy.universe.fetch_rates", side_effect=r), \
             patch("src.strategy.universe.fetch_candles", side_effect=c):
            uni = builder.build()
        self.assertEqual(uni.source_counts.get("seed"), 1)
        self.assertIn("seed", uni.reason_for_id[1001])

    def test_back_compat_base_count_llm_count_properties(self) -> None:
        self._record("AAPL", score=2.0)
        table = {"AAPL": 1001}
        cfg = UniverseConfig(max_tracked=5, enable_llm_rotation=False)
        builder = _make_builder(cfg, cache=self.cache, store=self.store)
        s, r, c = _resolve_table(table)
        with patch("src.strategy.universe.search_instrument", side_effect=s), \
             patch("src.strategy.universe.fetch_rates", side_effect=r), \
             patch("src.strategy.universe.fetch_candles", side_effect=c):
            uni = builder.build()
        self.assertEqual(uni.base_count, 1)  # news + seed + owned = news = 1
        self.assertEqual(uni.llm_count, 0)


if __name__ == "__main__":
    unittest.main()
