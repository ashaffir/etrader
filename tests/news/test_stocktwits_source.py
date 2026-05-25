"""Tests for the StockTwits trending-symbols source."""

import unittest

from src.news.sources.stocktwits import StockTwitsTrendingSource


_FAKE_RESPONSE = {
    "response": {"status": 200},
    "symbols": [
        {
            "id": 1,
            "symbol": "AAPL",
            "title": "Apple Inc.",
            "watchlist_count": 1500000,
            "instrument_class": "Stock",
        },
        {
            "id": 2,
            "symbol": "tsla",  # lowercase on purpose — should normalise
            "title": "Tesla Inc.",
            "watchlist_count": 980000,
            "instrument_class": "Stock",
        },
        {
            "id": 3,
            # Malformed — no symbol field. Should be skipped, not crash.
            "title": "broken row",
        },
    ],
}


class StockTwitsSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calls: list[str] = []

        def fake_fetcher(url: str) -> dict:
            self.calls.append(url)
            return _FAKE_RESPONSE

        self.fetcher = fake_fetcher
        self.source = StockTwitsTrendingSource(fetcher=fake_fetcher)

    def test_parses_each_ranked_symbol(self) -> None:
        items = list(self.source.fetch())
        symbols = [it.symbols[0] for it in items]
        self.assertEqual(symbols, ["AAPL", "TSLA"])

    def test_headline_includes_rank_and_watchers(self) -> None:
        items = list(self.source.fetch())
        first = items[0]
        self.assertIn("#1", first.headline)
        self.assertIn("$AAPL", first.headline)
        self.assertIn("watchers", first.headline)

    def test_metadata_carries_watchlist_count(self) -> None:
        items = list(self.source.fetch())
        self.assertEqual(items[0].metadata["watchlist_count"], 1500000)
        self.assertEqual(items[0].metadata["rank"], 1)

    def test_fetcher_failure_returns_empty(self) -> None:
        def boom(url: str) -> dict:
            raise RuntimeError("network down")

        source = StockTwitsTrendingSource(fetcher=boom)
        self.assertEqual(list(source.fetch()), [])

    def test_dedup_key_includes_source(self) -> None:
        items = list(self.source.fetch())
        self.assertTrue(items[0].dedup_key.startswith("stocktwits::"))


if __name__ == "__main__":
    unittest.main()
