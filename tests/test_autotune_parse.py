"""Unit tests for the LLM ``tuning`` block parser.

The parser is defensive on purpose — the LLM can return anything and
the bot must never crash on a malformed suggestion. These tests
exercise the happy path plus every malformed shape we anticipate.
"""

from __future__ import annotations

import unittest

from src.strategy.autotune_parse import parse_tune_request, render_tune_diff
from src.strategy.autotune_types import TuneApplied, TuneRequest


class ParseTuneRequestTests(unittest.TestCase):
    def test_parses_simple_threshold_change(self) -> None:
        payload = {
            "reason": "drought",
            "changes": [
                {"section": "strategy", "field": "min_signal_strength",
                 "value": 0.25, "rationale": "no candidates 60 cycles"},
            ],
        }
        req = parse_tune_request(payload)
        self.assertIsInstance(req, TuneRequest)
        self.assertEqual(req.reason, "drought")
        self.assertEqual(len(req.changes), 1)
        c = req.changes[0]
        self.assertEqual(c.section, "strategy")
        self.assertEqual(c.field, "min_signal_strength")
        self.assertAlmostEqual(c.value, 0.25)
        self.assertIn("no candidates", c.rationale)

    def test_coerces_int_field_from_float(self) -> None:
        req = parse_tune_request({
            "changes": [
                {"section": "strategy", "field": "rsi_period", "value": 14.0},
            ],
        })
        self.assertEqual(len(req.changes), 1)
        self.assertEqual(req.changes[0].value, 14)
        self.assertIsInstance(req.changes[0].value, int)

    def test_coerces_float_field_from_string(self) -> None:
        req = parse_tune_request({
            "changes": [
                {"section": "strategy", "field": "min_signal_strength",
                 "value": "0.30"},
            ],
        })
        self.assertEqual(len(req.changes), 1)
        self.assertAlmostEqual(req.changes[0].value, 0.30)

    def test_drops_unknown_section(self) -> None:
        req = parse_tune_request({
            "changes": [
                {"section": "guardrails", "field": "daily_loss_stop_usd",
                 "value": 9999.0},
            ],
        })
        self.assertEqual(len(req.changes), 0)

    def test_drops_unknown_field(self) -> None:
        req = parse_tune_request({
            "changes": [
                {"section": "strategy", "field": "min_signal_strenght",  # typo
                 "value": 0.30},
            ],
        })
        self.assertEqual(len(req.changes), 0)

    def test_drops_uncoercable_value(self) -> None:
        req = parse_tune_request({
            "changes": [
                {"section": "strategy", "field": "min_signal_strength",
                 "value": "not a number"},
            ],
        })
        self.assertEqual(len(req.changes), 0)

    def test_dedups_duplicate_section_field(self) -> None:
        req = parse_tune_request({
            "changes": [
                {"section": "strategy", "field": "min_signal_strength", "value": 0.20},
                {"section": "strategy", "field": "min_signal_strength", "value": 0.30},
            ],
        })
        self.assertEqual(len(req.changes), 1)
        self.assertAlmostEqual(req.changes[0].value, 0.20)

    def test_tools_section_spread_only(self) -> None:
        # spread_max_pct is the only tools field on the whitelist.
        req = parse_tune_request({
            "changes": [
                {"section": "tools", "field": "spread_max_pct", "value": 1.0},
                {"section": "tools", "field": "max_tools_per_cycle", "value": 20},
            ],
        })
        self.assertEqual(len(req.changes), 1)
        self.assertEqual(req.changes[0].field, "spread_max_pct")

    def test_non_dict_payload_returns_empty(self) -> None:
        for bad in (None, "tuning", 42, [], [{"changes": []}]):
            req = parse_tune_request(bad)
            self.assertTrue(req.is_empty, msg=f"failed on {bad!r}")

    def test_empty_changes_returns_empty_request(self) -> None:
        req = parse_tune_request({"changes": []})
        self.assertTrue(req.is_empty)

    def test_changes_not_a_list_returns_empty(self) -> None:
        req = parse_tune_request({"changes": "not a list"})
        self.assertTrue(req.is_empty)

    def test_change_entry_not_a_dict_skipped(self) -> None:
        req = parse_tune_request({
            "changes": [
                "not a dict",
                {"section": "strategy", "field": "min_signal_strength", "value": 0.3},
                42,
            ],
        })
        self.assertEqual(len(req.changes), 1)

    def test_render_diff_with_zero_changes(self) -> None:
        self.assertEqual(render_tune_diff([]), "(no-op)")

    def test_render_diff_multiple(self) -> None:
        applied = [
            TuneApplied(
                section="strategy", field="min_signal_strength",
                previous=0.40, current=0.30, rationale="r1",
            ),
            TuneApplied(
                section="tools", field="spread_max_pct",
                previous=0.5, current=1.0, rationale="r2",
            ),
        ]
        diff = render_tune_diff(applied)
        self.assertIn("strategy.min_signal_strength: 0.4", diff)
        self.assertIn("tools.spread_max_pct: 0.5", diff)
        self.assertIn(";", diff)


if __name__ == "__main__":
    unittest.main()
