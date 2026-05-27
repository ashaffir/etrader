"""Tests for the Telegram /directives and /tokens renderers."""

from __future__ import annotations

import unittest

from src.telegram_service.directives_formatter import (
    format_clear_result,
    format_directives,
    format_note_result,
    format_set_result,
)
from src.telegram_service.tokens_formatter import format_tokens


class DirectivesFormatterTests(unittest.TestCase):
    def test_disabled_payload_renders_unwired_message(self) -> None:
        out = format_directives({"enabled": False, "values": {}})
        self.assertIn("not wired", out)

    def test_all_disabled_defaults(self) -> None:
        out = format_directives({
            "enabled": True,
            "structured_keys": [
                "no_overnight", "hold_ceiling_minutes",
                "blocked_symbols", "blocked_sectors",
                "max_total_account_invested_usd",
            ],
            "values": {
                "no_overnight": False,
                "hold_ceiling_minutes": 0,
                "blocked_symbols": [],
                "blocked_sectors": [],
                "max_total_account_invested_usd": 0.0,
                "notes": "",
            },
        })
        self.assertIn("[DIRECTIVES]", out)
        self.assertIn("no_overnight", out)
        self.assertIn("false", out)
        self.assertIn("(disabled)", out)
        self.assertIn("(none)", out)
        self.assertIn("notes: (none)", out)
        self.assertIn("/directive set", out)

    def test_active_directives_render(self) -> None:
        out = format_directives({
            "enabled": True,
            "values": {
                "no_overnight": True,
                "hold_ceiling_minutes": 120,
                "blocked_symbols": ["NVDA", "TSLA"],
                "blocked_sectors": ["Energy"],
                "max_total_account_invested_usd": 3000.5,
                "notes": "prefer financials this week",
            },
        })
        self.assertIn("true", out)
        self.assertIn("120 min", out)
        self.assertIn("NVDA", out)
        self.assertIn("TSLA", out)
        self.assertIn("Energy", out)
        self.assertIn("$3000.50", out)
        self.assertIn("prefer financials", out)

    def test_set_result_shows_before_and_after(self) -> None:
        out = format_set_result({
            "key": "no_overnight",
            "previous": False,
            "current": True,
        })
        self.assertIn("no_overnight", out)
        self.assertIn("before", out)
        self.assertIn("after", out)
        self.assertIn("true", out)
        self.assertIn("false", out)

    def test_clear_result_shows_default(self) -> None:
        out = format_clear_result({
            "key": "hold_ceiling_minutes",
            "previous": 120,
            "current": 0,
        })
        self.assertIn("Cleared", out)
        self.assertIn("hold_ceiling_minutes", out)
        self.assertIn("120", out)

    def test_note_set_with_payload(self) -> None:
        out = format_note_result({
            "previous": "",
            "current": "hello world",
        })
        self.assertIn("hello world", out)

    def test_note_cleared_explicit(self) -> None:
        out = format_note_result(
            {"previous": "abc", "current": ""}, cleared=True,
        )
        self.assertIn("Notes cleared", out)
        self.assertIn("3 chars", out)


class TokensFormatterTests(unittest.TestCase):
    def test_disabled_payload(self) -> None:
        out = format_tokens({"enabled": False})
        self.assertIn("disabled", out)
        self.assertIn("AZURE_OPENAI", out)

    def test_full_payload_renders_windows_and_call_types(self) -> None:
        out = format_tokens({
            "enabled": True,
            "deployment": "gpt-5-mini-prod",
            "rates": {
                "family": "gpt-5-mini",
                "input_per_m": 0.25,
                "cached_per_m": 0.03,
                "output_per_m": 2.00,
            },
            "today": {
                "calls": 5, "prompt_tokens": 12345, "cached_tokens": 100,
                "completion_tokens": 678, "cost_usd": 0.04,
            },
            "last_7d": {
                "calls": 30, "prompt_tokens": 99000, "cached_tokens": 0,
                "completion_tokens": 5000, "cost_usd": 0.55,
            },
            "all_time": {
                "calls": 100, "prompt_tokens": 500000, "cached_tokens": 0,
                "completion_tokens": 20000, "cost_usd": 3.21,
            },
            "by_call_type": {
                "decision": {
                    "calls": 3, "prompt_tokens": 6000, "cached_tokens": 0,
                    "completion_tokens": 300, "cost_usd": 0.02,
                },
                "qa": {
                    "calls": 2, "prompt_tokens": 6345, "cached_tokens": 0,
                    "completion_tokens": 378, "cost_usd": 0.02,
                },
            },
            "last_call": {
                "call_type": "decision",
                "timestamp": "2026-05-27T07:15:00Z",
                "prompt_tokens": 1200,
                "completion_tokens": 60,
                "cost_usd": 0.003,
            },
        })
        self.assertIn("[TOKENS]", out)
        self.assertIn("gpt-5-mini-prod", out)
        self.assertIn("TODAY", out)
        self.assertIn("7 DAYS", out)
        self.assertIn("ALL-TIME", out)
        self.assertIn("decision", out)
        self.assertIn("qa", out)
        self.assertIn("LAST CALL", out)

    def test_missing_rates_renders_dash(self) -> None:
        out = format_tokens({
            "enabled": True,
            "deployment": "unknown-deployment",
            "rates": None,
            "today": {
                "calls": 0, "prompt_tokens": 0, "cached_tokens": 0,
                "completion_tokens": 0, "cost_usd": 0,
            },
        })
        self.assertIn("—", out)
        self.assertIn("no entry", out)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
