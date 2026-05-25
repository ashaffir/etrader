"""Tests for the universe + news Telegram formatters."""

import unittest

from src.telegram_service.formatters import format_news, format_universe


class FormatUniverseTests(unittest.TestCase):
    def test_renders_symbols_with_reasons(self) -> None:
        text = format_universe({
            "symbols": ["AAPL", "MSFT"],
            "source_counts": {"news": 2},
            "reasons": {
                "AAPL": "[stocktwits] trending #3",
                "MSFT": "[yfinance] beats Q3",
            },
            "rejected": {},
        })
        self.assertIn("[UNIVERSE]", text)
        self.assertIn("tracking 2 instrument(s)", text)
        self.assertIn("news=2", text)
        self.assertIn("AAPL: [stocktwits] trending #3", text)
        self.assertIn("MSFT: [yfinance] beats Q3", text)

    def test_empty_universe_lists_rejections(self) -> None:
        text = format_universe({
            "symbols": [],
            "source_counts": {},
            "reasons": {},
            "rejected": {"AAPL": "flat", "MSFT": "wide spread"},
        })
        self.assertIn("no instruments tracked", text)
        self.assertIn("Recent rejections", text)
        self.assertIn("AAPL: flat", text)
        self.assertIn("MSFT: wide spread", text)

    def test_caps_displayed_rejections(self) -> None:
        rejected = {f"SYM{i}": f"reason{i}" for i in range(15)}
        text = format_universe({
            "symbols": ["AAPL"],
            "source_counts": {"news": 1},
            "reasons": {"AAPL": "[stocktwits] trending"},
            "rejected": rejected,
        })
        self.assertIn("Rejected this refresh (15)", text)
        self.assertIn("5 more", text)


class FormatNewsTests(unittest.TestCase):
    def test_renders_candidates(self) -> None:
        text = format_news({
            "candidates": [
                {
                    "symbol": "AAPL",
                    "score": 1.25,
                    "sources": ["stocktwits", "yfinance"],
                    "headlines": ["Apple beats Q3 estimates"],
                    "first_seen_unix": 0,
                    "last_seen_unix": 0,
                    "reason": "[stocktwits+yfinance] Apple beats Q3 estimates",
                },
            ],
            "last_scan": {
                "started_at_unix": 0,
                "finished_at_unix": 1716580000,
                "items_fetched": 80,
                "items_kept": 75,
                "observations_recorded": 100,
                "per_source_counts": {"stocktwits": 30, "yfinance": 25, "google_news": 20},
                "per_source_errors": {},
            },
            "next_scan_in_seconds": 1800.0,
        })
        self.assertIn("[NEWS]", text)
        self.assertIn("AAPL", text)
        self.assertIn("score= 1.25", text)
        self.assertIn("stocktwits+yfinance", text)
        self.assertIn("Apple beats Q3 estimates", text)
        self.assertIn("stocktwits=30", text)
        self.assertIn("next scan in: ~30 min", text)

    def test_empty_store_with_no_scan_yet(self) -> None:
        text = format_news({
            "candidates": [],
            "last_scan": None,
            "next_scan_in_seconds": None,
        })
        self.assertIn("(none yet)", text)
        self.assertIn("(candidate store empty)", text)

    def test_renders_scan_errors(self) -> None:
        text = format_news({
            "candidates": [],
            "last_scan": {
                "started_at_unix": 0,
                "finished_at_unix": 1716580000,
                "items_fetched": 0,
                "items_kept": 0,
                "observations_recorded": 0,
                "per_source_counts": {},
                "per_source_errors": {"stocktwits": "rate limited"},
            },
            "next_scan_in_seconds": 0,
        })
        self.assertIn("errors:", text)
        self.assertIn("stocktwits", text)


if __name__ == "__main__":
    unittest.main()
