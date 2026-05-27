"""Tests for src/ai/usage_tracker.py."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.ai.pricing import TokenRates
from src.ai.usage_tracker import LLMUsageTracker


def _fixed_rates() -> TokenRates:
    return TokenRates("test", input_per_m=1.0, cached_per_m=0.1, output_per_m=4.0)


class UsageTrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "llm_usage.jsonl"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _tracker(self) -> LLMUsageTracker:
        return LLMUsageTracker(
            self.path, deployment="test", rates_override=_fixed_rates(),
        )

    def test_record_emits_jsonl_line(self) -> None:
        tracker = self._tracker()
        tracker.record(
            prompt_tokens=1000, completion_tokens=500, cached_tokens=100,
            call_type="decision",
        )
        lines = self.path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        entry = json.loads(lines[0])
        self.assertEqual(entry["prompt_tokens"], 1000)
        self.assertEqual(entry["call_type"], "decision")
        self.assertIn("cost_usd", entry)

    def test_snapshot_accumulates_today_and_all_time(self) -> None:
        tracker = self._tracker()
        tracker.record(prompt_tokens=1000, completion_tokens=200, call_type="decision")
        tracker.record(prompt_tokens=500, completion_tokens=100, call_type="qa")
        snap = tracker.snapshot()
        self.assertTrue(snap.get("enabled") in (None, True))  # set by controller
        self.assertEqual(snap["today"]["calls"], 2)
        self.assertEqual(snap["today"]["prompt_tokens"], 1500)
        self.assertEqual(snap["today"]["completion_tokens"], 300)
        self.assertEqual(snap["all_time"]["calls"], 2)
        self.assertIn("decision", snap["by_call_type"])
        self.assertIn("qa", snap["by_call_type"])

    def test_persist_then_restore(self) -> None:
        first = self._tracker()
        first.record(prompt_tokens=1000, completion_tokens=200, call_type="decision")
        first.record(prompt_tokens=500, completion_tokens=100, call_type="qa")

        second = self._tracker()
        snap = second.snapshot()
        # Today rolls cleanly across the restart (same UTC day).
        self.assertEqual(snap["all_time"]["calls"], 2)
        self.assertEqual(snap["all_time"]["prompt_tokens"], 1500)

    def test_cost_uses_rates(self) -> None:
        tracker = self._tracker()
        tracker.record(prompt_tokens=1_000_000, completion_tokens=0, call_type="decision")
        snap = tracker.snapshot()
        self.assertAlmostEqual(snap["today"]["cost_usd"], 1.0, places=6)

    def test_day_roll_resets_call_type_today(self) -> None:
        tracker = self._tracker()
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        tracker.record(
            prompt_tokens=100, completion_tokens=10, call_type="decision",
            timestamp=yesterday,
        )
        tracker.record(
            prompt_tokens=200, completion_tokens=20, call_type="qa",
        )
        snap = tracker.snapshot()
        # yesterday's "decision" call must NOT appear in by_call_type
        # because that bucket scopes to "today" only.
        self.assertNotIn("decision", snap["by_call_type"])
        self.assertIn("qa", snap["by_call_type"])
        # all_time still includes both.
        self.assertEqual(snap["all_time"]["calls"], 2)

    def test_record_returns_entry(self) -> None:
        tracker = self._tracker()
        entry = tracker.record(
            prompt_tokens=1, completion_tokens=2, call_type="decision",
        )
        self.assertEqual(entry.prompt_tokens, 1)
        self.assertEqual(entry.completion_tokens, 2)
        self.assertEqual(entry.call_type, "decision")
        self.assertGreaterEqual(entry.cost_usd, 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
