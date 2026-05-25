"""Unit tests for :class:`YFinanceFundamentalsFetcher`.

We mock the ``info_fetcher`` callable so the test runs without
yfinance / network access.
"""

import math
import unittest

from src.fundamentals.yfinance_fetcher import YFinanceFundamentalsFetcher


_INFO_AAPL = {
    "symbol": "AAPL",
    "shortName": "Apple Inc.",
    "exchange": "NMS",
    "quoteType": "EQUITY",
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "country": "United States",
    "currency": "usd",
    "marketCap": 4.5e12,
    "enterpriseValue": 4.55e12,
    "trailingPE": 37.43,
    "forwardPE": 32.16,
    "priceToBook": 42.53,
    "priceToSalesTrailing12Months": 10.05,
    "dividendYield": 0.35,
    "beta": 1.07,
    "profitMargins": 0.27,
    "operatingMargins": 0.32,
    "returnOnEquity": 1.41,
    "revenueGrowth": 0.16,
    "earningsGrowth": 0.21,
    "debtToEquity": 79.5,
    "fiftyTwoWeekHigh": 311.4,
    "fiftyTwoWeekLow": 195.07,
    "averageVolume10days": 43_241_680,
    "targetMeanPrice": 308.6,
    "recommendationKey": "Buy",  # mixed case → lowercased
    "numberOfAnalystOpinions": 43,
    "earningsTimestamp": 1_777_579_200,
    "longBusinessSummary": "Apple designs, manufactures, and markets " * 20,
    "website": "https://apple.com",
    "fullTimeEmployees": 160_000,
    "sharesOutstanding": 14.7e9,
    "floatShares": 14.6e9,
    "heldPercentInsiders": 0.016,
    "heldPercentInstitutions": 0.658,
}


class YFinanceFetcherTests(unittest.TestCase):
    def test_maps_full_payload_to_snapshot(self) -> None:
        clock = lambda: 1_700_000_000.0  # noqa: E731 — tiny test helper
        f = YFinanceFundamentalsFetcher(
            info_fetcher=lambda sym: _INFO_AAPL,
            clock=clock,
        )
        snap = f.fetch("aapl")
        assert snap is not None  # type: ignore[unreachable]
        self.assertEqual(snap.symbol, "AAPL")
        self.assertAlmostEqual(snap.fetched_at_unix, 1_700_000_000.0)
        self.assertEqual(snap.name, "Apple Inc.")
        self.assertEqual(snap.sector, "Technology")
        self.assertEqual(snap.industry, "Consumer Electronics")
        self.assertEqual(snap.quote_type, "EQUITY")
        self.assertEqual(snap.currency, "USD")  # uppercased
        self.assertEqual(snap.analyst_recommendation, "buy")  # lowercased
        self.assertEqual(snap.analyst_count, 43)
        self.assertAlmostEqual(snap.trailing_pe, 37.43)
        self.assertAlmostEqual(snap.dividend_yield, 0.35)
        self.assertAlmostEqual(snap.fifty_two_week_high, 311.4)
        self.assertIsNotNone(snap.summary)
        self.assertLessEqual(len(snap.summary or ""), 240)
        # Extras are populated but capped.
        self.assertIn("website", snap.extras)
        self.assertIn("sharesOutstanding", snap.extras)

    def test_missing_fields_are_none(self) -> None:
        f = YFinanceFundamentalsFetcher(info_fetcher=lambda sym: {"shortName": "ETH-USD"})
        snap = f.fetch("ETH-USD")
        assert snap is not None  # type: ignore[unreachable]
        self.assertEqual(snap.name, "ETH-USD")
        self.assertIsNone(snap.trailing_pe)
        self.assertIsNone(snap.profit_margin)
        self.assertEqual(snap.extras, {})

    def test_nan_coerces_to_none(self) -> None:
        f = YFinanceFundamentalsFetcher(
            info_fetcher=lambda sym: {"shortName": "X", "trailingPE": math.nan},
        )
        snap = f.fetch("X")
        assert snap is not None  # type: ignore[unreachable]
        self.assertIsNone(snap.trailing_pe)

    def test_provider_error_returns_none(self) -> None:
        def boom(_sym):
            raise RuntimeError("yahoo blocked")
        f = YFinanceFundamentalsFetcher(info_fetcher=boom)
        self.assertIsNone(f.fetch("AAPL"))

    def test_empty_info_returns_none(self) -> None:
        f = YFinanceFundamentalsFetcher(info_fetcher=lambda sym: {})
        self.assertIsNone(f.fetch("AAPL"))

    def test_blank_symbol_returns_none(self) -> None:
        f = YFinanceFundamentalsFetcher(info_fetcher=lambda sym: _INFO_AAPL)
        self.assertIsNone(f.fetch(""))
        self.assertIsNone(f.fetch("   "))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
