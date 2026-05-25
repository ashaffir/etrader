"""Tests for the persistent candidate store."""

import json
import tempfile
import unittest
from pathlib import Path

from src.news.candidate_store import MAX_HEADLINES, CandidateStore


class CandidateStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.path = self.tmp / "candidates.json"
        # Frozen clock for deterministic TTL behaviour
        self.now = 1000.0
        self.store = CandidateStore(
            path=self.path,
            ttl_seconds=600,
            clock=lambda: self.now,
        )

    def test_record_creates_candidate(self) -> None:
        cand = self.store.record(
            symbol="aapl", source="stocktwits", headline="trending", weight=1.0
        )
        self.assertEqual(cand.symbol, "AAPL")
        self.assertEqual(cand.score, 1.0)
        self.assertEqual(cand.sources, ["stocktwits"])
        self.assertEqual(cand.headlines, ["trending"])
        self.assertEqual(cand.first_seen_unix, 1000.0)
        self.assertEqual(cand.last_seen_unix, 1000.0)

    def test_record_accumulates_score_and_sources(self) -> None:
        self.store.record(symbol="AAPL", source="stocktwits", headline="h1", weight=0.7)
        self.now += 30
        self.store.record(symbol="AAPL", source="yfinance", headline="h2", weight=0.5)
        self.now += 30
        # Same source again — source set stays unique, score still bumps.
        self.store.record(symbol="AAPL", source="stocktwits", headline="h3", weight=0.4)

        cand = self.store.top()[0]
        self.assertAlmostEqual(cand.score, 0.7 + 0.5 + 0.4, places=6)
        self.assertEqual(set(cand.sources), {"stocktwits", "yfinance"})
        self.assertEqual(cand.headlines, ["h3", "h2", "h1"])

    def test_headlines_capped_at_max(self) -> None:
        for i in range(MAX_HEADLINES + 3):
            self.store.record(symbol="AAPL", source="src", headline=f"h{i}")
        cand = self.store.top()[0]
        self.assertEqual(len(cand.headlines), MAX_HEADLINES)
        # Newest first.
        self.assertEqual(cand.headlines[0], f"h{MAX_HEADLINES + 2}")

    def test_duplicate_headline_not_added_twice(self) -> None:
        self.store.record(symbol="AAPL", source="src", headline="same")
        self.store.record(symbol="AAPL", source="src", headline="same")
        self.assertEqual(self.store.top()[0].headlines, ["same"])

    def test_extend_bulk(self) -> None:
        applied = self.store.extend(
            [
                {"symbol": "AAPL", "source": "stocktwits", "headline": "a"},
                {"symbol": "MSFT", "source": "yfinance", "headline": "m", "weight": 0.5},
                # Malformed rows are skipped, not raised.
                {"symbol": 123, "source": "x", "headline": "y"},
                {"source": "x", "headline": "y"},  # missing symbol
            ]
        )
        self.assertEqual(applied, 2)
        self.assertEqual(len(self.store), 2)

    def test_top_sorted_by_score_descending(self) -> None:
        self.store.record(symbol="AAPL", source="s", headline="", weight=0.3)
        self.store.record(symbol="MSFT", source="s", headline="", weight=0.9)
        self.store.record(symbol="NVDA", source="s", headline="", weight=0.6)
        ordered = [c.symbol for c in self.store.top()]
        self.assertEqual(ordered, ["MSFT", "NVDA", "AAPL"])

    def test_top_limit(self) -> None:
        for sym, w in [("AAPL", 0.3), ("MSFT", 0.9), ("NVDA", 0.6)]:
            self.store.record(symbol=sym, source="s", headline="", weight=w)
        self.assertEqual([c.symbol for c in self.store.top(2)], ["MSFT", "NVDA"])

    def test_prune_drops_stale(self) -> None:
        self.store.record(symbol="AAPL", source="s", headline="")
        self.now += 1000  # > 600 s TTL
        self.store.record(symbol="MSFT", source="s", headline="")
        removed = self.store.prune()
        self.assertEqual(removed, 1)
        self.assertNotIn("AAPL", self.store)
        self.assertIn("MSFT", self.store)

    def test_round_trip_persistence(self) -> None:
        self.store.record(symbol="AAPL", source="src", headline="h", weight=0.8)
        self.store.save()
        # Reload from disk
        again = CandidateStore(
            path=self.path,
            ttl_seconds=600,
            clock=lambda: self.now,
        )
        self.assertIn("AAPL", again)
        cand = again.top()[0]
        self.assertEqual(cand.symbol, "AAPL")
        self.assertAlmostEqual(cand.score, 0.8, places=6)

    def test_corrupt_file_does_not_crash(self) -> None:
        self.path.write_text("{not json", encoding="utf-8")
        store = CandidateStore(
            path=self.path, ttl_seconds=600, clock=lambda: self.now
        )
        self.assertEqual(len(store), 0)

    def test_record_rejects_empty_symbol(self) -> None:
        with self.assertRaises(ValueError):
            self.store.record(symbol="   ", source="s", headline="x")

    def test_reason_string_includes_source_and_headline(self) -> None:
        self.store.record(symbol="AAPL", source="stocktwits", headline="trending up")
        self.assertIn("stocktwits", self.store.top()[0].reason)
        self.assertIn("trending up", self.store.top()[0].reason)

    def test_save_written_as_pretty_json(self) -> None:
        self.store.record(symbol="AAPL", source="s", headline="h", weight=1.0)
        self.store.save()
        body = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertIn("candidates", body)
        self.assertIn("AAPL", body["candidates"])


if __name__ == "__main__":
    unittest.main()
