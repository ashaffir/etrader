"""Tests for the news aggregator orchestration layer."""

import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

from src.news.aggregator import NewsAggregator
from src.news.candidate_store import CandidateStore
from src.news.sources.base import NewsItem
from src.news.ticker_extractor import TickerExtractor


class _FakeSource:
    """Test double matching the NewsSource Protocol."""

    def __init__(self, name: str, items: list[NewsItem]):
        self.name = name
        self._items = items
        self.calls: list[dict[str, Any]] = []

    def fetch(self, *, since=None, known_symbols=None):
        self.calls.append({"since": since, "known_symbols": list(known_symbols or [])})
        return list(self._items)


class _RaisingSource:
    name = "boom"

    def fetch(self, *, since=None, known_symbols=None):
        raise RuntimeError("forced failure")


class NewsAggregatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.now = 1_700_000_000.0
        self.store = CandidateStore(
            path=self.tmp / "cands.json",
            ttl_seconds=24 * 3600,
            clock=lambda: self.now,
        )
        self.extractor = TickerExtractor(known_symbols=["AAPL", "MSFT", "TSLA"])

    def _agg(self, sources) -> NewsAggregator:
        return NewsAggregator(
            sources=sources,
            store=self.store,
            ticker_extractor=self.extractor,
            clock=lambda: self.now,
        )

    def test_pre_tagged_source_records_without_extraction(self) -> None:
        item = NewsItem(
            source="stocktwits",
            symbols=("AAPL",),
            headline="trending",
            url="https://x/1",
            published_at=self.now,
        )
        agg = self._agg([_FakeSource("stocktwits", [item])])
        stats = agg.run()
        self.assertEqual(stats.items_kept, 1)
        self.assertEqual(stats.observations_recorded, 1)
        self.assertIn("AAPL", self.store)

    def test_extraction_kicks_in_for_untagged_items(self) -> None:
        item = NewsItem(
            source="google_news",
            symbols=(),
            headline="AAPL beats on earnings; TSLA lags",
            url="https://x/2",
            published_at=self.now,
        )
        agg = self._agg([_FakeSource("google_news", [item])])
        agg.run()
        symbols = {c.symbol for c in self.store.top()}
        self.assertEqual(symbols, {"AAPL", "TSLA"})

    def test_dedup_across_sources_by_url(self) -> None:
        same_url = "https://x/dup"
        a = NewsItem(
            source="google_news",
            symbols=("AAPL",),
            headline="X",
            url=same_url,
            published_at=self.now,
        )
        # Same source+url should be deduped within the run.
        agg = self._agg([_FakeSource("google_news", [a, a])])
        stats = agg.run()
        self.assertEqual(stats.items_kept, 1)

    def test_raising_source_does_not_kill_run(self) -> None:
        good = NewsItem(
            source="stocktwits",
            symbols=("AAPL",),
            headline="x",
            url="https://x/g",
            published_at=self.now,
        )
        agg = self._agg([_RaisingSource(), _FakeSource("stocktwits", [good])])
        stats = agg.run()
        self.assertIn("boom", stats.per_source_errors)
        self.assertIn("stocktwits", stats.per_source_counts)
        self.assertIn("AAPL", self.store)

    def test_freshness_decay_lowers_weight(self) -> None:
        recent = NewsItem(
            source="stocktwits",
            symbols=("AAPL",),
            headline="recent",
            url="https://x/r",
            published_at=self.now,
        )
        # 24h old → 4 half-lives at 6h half-life → 1/16 weight.
        old = NewsItem(
            source="stocktwits",
            symbols=("MSFT",),
            headline="old",
            url="https://x/o",
            published_at=self.now - 24 * 3600,
        )
        agg = self._agg([_FakeSource("stocktwits", [recent, old])])
        agg.run()
        cands = {c.symbol: c.score for c in self.store.top()}
        self.assertGreater(cands["AAPL"], cands["MSFT"] * 10)

    def test_known_symbols_passed_to_sources(self) -> None:
        src = _FakeSource("yfinance", [])
        agg = self._agg([src])
        agg.run(known_symbols=["AAPL", "MSFT"])
        self.assertEqual(src.calls[0]["known_symbols"], ["AAPL", "MSFT"])

    def test_run_stats_counts(self) -> None:
        items = [
            NewsItem(
                source="stocktwits",
                symbols=("AAPL",),
                headline=f"h{i}",
                url=f"https://x/{i}",
                published_at=self.now,
            )
            for i in range(3)
        ]
        agg = self._agg([_FakeSource("stocktwits", items)])
        stats = agg.run()
        self.assertEqual(stats.items_fetched, 3)
        self.assertEqual(stats.items_kept, 3)
        self.assertEqual(stats.observations_recorded, 3)
        self.assertGreaterEqual(stats.finished_at_unix, stats.started_at_unix)

    def test_persistence_after_run(self) -> None:
        item = NewsItem(
            source="stocktwits",
            symbols=("AAPL",),
            headline="x",
            url="https://x/p",
            published_at=self.now,
        )
        agg = self._agg([_FakeSource("stocktwits", [item])])
        agg.run()
        # Reload the store from disk — record should persist.
        again = CandidateStore(
            path=self.tmp / "cands.json",
            ttl_seconds=24 * 3600,
            clock=lambda: self.now,
        )
        self.assertIn("AAPL", again)


if __name__ == "__main__":
    unittest.main()
