"""Tests for the yfinance per-symbol news source."""

import unittest

from src.news.sources.yfinance_news import YFinanceNewsSource


_FAKE_NEWS = {
    "AAPL": [
        {
            "title": "Apple beats Q3 estimates",
            "link": "https://news/apple-q3",
            "providerPublishTime": 1716580000,
            "publisher": "Reuters",
            "type": "STORY",
            "relatedTickers": ["MSFT", "GOOGL"],
        },
        {
            # missing link → must be skipped
            "title": "broken row",
            "providerPublishTime": 1716580000,
        },
    ],
}


class YFinanceNewsSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calls: list[str] = []

        def fake(symbol: str):
            self.calls.append(symbol)
            return _FAKE_NEWS.get(symbol, [])

        self.fetcher = fake
        self.source = YFinanceNewsSource(fetcher=fake)

    def test_returns_empty_when_no_known_symbols(self) -> None:
        self.assertEqual(list(self.source.fetch()), [])

    def test_emits_item_with_related_tickers_carried_through(self) -> None:
        items = list(self.source.fetch(known_symbols=["AAPL"]))
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual(item.headline, "Apple beats Q3 estimates")
        # Queried symbol first, related tickers after, all upper, unique.
        self.assertEqual(item.symbols, ("AAPL", "MSFT", "GOOGL"))
        self.assertEqual(item.metadata["publisher"], "Reuters")
        self.assertEqual(item.metadata["queried_symbol"], "AAPL")

    def test_caps_max_symbols(self) -> None:
        source = YFinanceNewsSource(fetcher=self.fetcher, max_symbols=2)
        list(source.fetch(known_symbols=["AAPL", "MSFT", "NVDA", "AMD"]))
        self.assertEqual(self.calls, ["AAPL", "MSFT"])

    def test_dedup_known_symbols_case_insensitive(self) -> None:
        list(self.source.fetch(known_symbols=["AAPL", "aapl", "  AAPL  "]))
        self.assertEqual(self.calls, ["AAPL"])

    def test_failing_symbol_does_not_kill_run(self) -> None:
        def fake(symbol: str):
            if symbol == "BROKEN":
                raise RuntimeError("rate limited")
            return _FAKE_NEWS.get(symbol, [])

        source = YFinanceNewsSource(fetcher=fake)
        items = list(source.fetch(known_symbols=["BROKEN", "AAPL"]))
        # AAPL's item must still come through.
        self.assertEqual(len(items), 1)

    def test_since_cutoff(self) -> None:
        items = list(self.source.fetch(since=2_000_000_000.0, known_symbols=["AAPL"]))
        # Fake item is from 2024, well before cutoff → filtered.
        self.assertEqual(items, [])


if __name__ == "__main__":
    unittest.main()
