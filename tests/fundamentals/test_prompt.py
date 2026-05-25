"""Tests for the LLM-payload projection of fundamentals snapshots."""

import unittest

from src.fundamentals.prompt import build_fundamentals_payload, project_for_llm
from src.fundamentals.types import FundamentalsSnapshot


def _snap(symbol: str = "AAPL") -> FundamentalsSnapshot:
    return FundamentalsSnapshot(
        symbol=symbol,
        fetched_at_unix=1_700_000_000.0,
        name="Apple Inc.",
        sector="Technology",
        industry="Consumer Electronics",
        quote_type="EQUITY",
        currency="USD",
        market_cap=4.5e12,
        trailing_pe=37.4,
        forward_pe=32.0,
        price_to_book=42.5,
        price_to_sales=10.0,
        dividend_yield=0.35,
        beta=1.07,
        profit_margin=0.27,
        operating_margin=0.32,
        return_on_equity=1.41,
        revenue_growth=0.16,
        earnings_growth=0.21,
        debt_to_equity=79.5,
        fifty_two_week_high=311.4,
        fifty_two_week_low=195.07,
        analyst_target_mean=308.6,
        analyst_recommendation="buy",
        analyst_count=43,
        summary="Apple designs, manufactures, and markets smartphones.",
        extras={"website": "https://apple.com", "fullTimeEmployees": 160_000},
    )


class ProjectForLLMTests(unittest.TestCase):
    def test_emits_only_curated_keys(self) -> None:
        out = project_for_llm(_snap())
        # Curated set — must include these:
        for key in (
            "symbol", "name", "sector", "industry", "quote_type", "currency",
            "market_cap", "trailing_pe", "forward_pe", "price_to_book",
            "price_to_sales", "dividend_yield", "beta",
            "profit_margin", "operating_margin", "return_on_equity",
            "revenue_growth", "earnings_growth", "debt_to_equity",
            "fifty_two_week_high", "fifty_two_week_low",
            "analyst_target_mean", "analyst_recommendation", "analyst_count",
            "summary",
        ):
            self.assertIn(key, out, f"missing curated key {key}")
        # Bulky / low-signal fields are NOT projected:
        for key in ("extras", "fetched_at_unix", "next_earnings_unix", "source"):
            self.assertNotIn(key, out, f"unexpected key {key} in LLM projection")

    def test_handles_missing_values(self) -> None:
        snap = FundamentalsSnapshot(symbol="BTC-USD", fetched_at_unix=0.0)
        out = project_for_llm(snap)
        self.assertEqual(out["symbol"], "BTC-USD")
        self.assertIsNone(out["trailing_pe"])
        self.assertIsNone(out["sector"])

    def test_build_payload_skips_none(self) -> None:
        payload = build_fundamentals_payload([_snap(), None, _snap("MSFT")])  # type: ignore[list-item]
        self.assertEqual(len(payload), 2)
        self.assertEqual([p["symbol"] for p in payload], ["AAPL", "MSFT"])

    def test_build_payload_accepts_mapping(self) -> None:
        payload = build_fundamentals_payload({"AAPL": _snap("AAPL"), "MSFT": _snap("MSFT")})
        self.assertEqual({p["symbol"] for p in payload}, {"AAPL", "MSFT"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
