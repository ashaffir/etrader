"""Tests for the /fundamentals Telegram formatter."""

import unittest

from src.telegram_service.formatters import format_fundamentals


_DETAIL_PAYLOAD = {
    "enabled": True,
    "symbol": "AAPL",
    "snapshot": {
        "symbol": "AAPL",
        "fetched_at_unix": 1_700_000_000.0,
        "name": "Apple Inc.",
        "exchange": "NMS",
        "quote_type": "EQUITY",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "country": "United States",
        "currency": "USD",
        "market_cap": 4.5e12,
        "enterprise_value": 4.55e12,
        "trailing_pe": 37.43,
        "forward_pe": 32.16,
        "price_to_book": 42.53,
        "price_to_sales": 10.05,
        "dividend_yield": 0.35,
        "beta": 1.07,
        "profit_margin": 0.27,
        "operating_margin": 0.32,
        "return_on_equity": 1.41,
        "revenue_growth": 0.16,
        "earnings_growth": 0.21,
        "debt_to_equity": 79.5,
        "fifty_two_week_high": 311.4,
        "fifty_two_week_low": 195.07,
        "analyst_target_mean": 308.6,
        "analyst_recommendation": "buy",
        "analyst_count": 43,
        "next_earnings_unix": 1_777_579_200.0,
        "summary": "Apple designs, manufactures, and markets smartphones.",
        "extras": {},
    },
}


class FundamentalsFormatterTests(unittest.TestCase):
    def test_disabled_payload(self) -> None:
        out = format_fundamentals({"enabled": False, "symbol": None, "snapshot": None, "items": []})
        self.assertIn("disabled", out)

    def test_list_view(self) -> None:
        out = format_fundamentals({
            "enabled": True,
            "count": 2,
            "items": [
                {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology", "fetched_at_unix": 1_700_000_000.0},
                {"symbol": "BTC-USD", "name": "Bitcoin", "sector": None, "fetched_at_unix": 1_700_000_000.0},
            ],
        })
        self.assertIn("FUNDAMENTALS", out)
        self.assertIn("Technology:", out)
        self.assertIn("AAPL", out)
        self.assertIn("(no sector)", out)
        self.assertIn("BTC-USD", out)

    def test_list_view_empty(self) -> None:
        out = format_fundamentals({"enabled": True, "count": 0, "items": []})
        self.assertIn("cache empty", out)

    def test_detail_view_has_sections(self) -> None:
        out = format_fundamentals(_DETAIL_PAYLOAD)
        self.assertIn("AAPL", out)
        self.assertIn("Apple Inc.", out)
        self.assertIn("Valuation:", out)
        self.assertIn("Profitability / growth:", out)
        self.assertIn("Analyst consensus:", out)
        self.assertIn("Next earnings:", out)
        self.assertIn("4.50T", out)  # market_cap rendered compactly

    def test_detail_unknown_symbol(self) -> None:
        out = format_fundamentals({"enabled": True, "symbol": "ZZZ", "snapshot": None})
        self.assertIn("ZZZ", out)
        self.assertIn("not cached", out)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
