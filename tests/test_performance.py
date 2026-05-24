"""ToolPerformanceLog: settle correctness + JSONL persistence."""

import json
import tempfile
import unittest
from pathlib import Path

from src.strategy.performance import ToolPerformanceLog


class PerformanceLogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "perf.jsonl"
        self.log = ToolPerformanceLog(path=self.path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_settle_records_hit_when_score_matches_return(self) -> None:
        self.log.record_scores(cycle_index=1, instrument_id=10, scores={"sma_cross": 0.5})
        self.log.record_close(cycle_index=1, instrument_id=10, close=100.0)
        # Cycle 2 close is higher → positive return → score>0 hit.
        self.log.record_close(cycle_index=2, instrument_id=10, close=110.0)
        self.log.settle(current_cycle=2)
        stats = self.log.stats_for("sma_cross")
        self.assertEqual(stats.observations, 1)
        self.assertEqual(stats.hits, 1)
        self.assertAlmostEqual(stats.hit_rate, 1.0)

    def test_settle_records_miss_when_score_opposes_return(self) -> None:
        self.log.record_scores(cycle_index=1, instrument_id=10, scores={"rsi": 0.5})
        self.log.record_close(cycle_index=1, instrument_id=10, close=100.0)
        self.log.record_close(cycle_index=2, instrument_id=10, close=90.0)
        self.log.settle(current_cycle=2)
        stats = self.log.stats_for("rsi")
        self.assertEqual(stats.misses, 1)

    def test_jsonl_persisted(self) -> None:
        self.log.record_scores(cycle_index=1, instrument_id=10, scores={"sma_cross": 0.5})
        self.log.record_close(cycle_index=1, instrument_id=10, close=100.0)
        self.log.record_close(cycle_index=2, instrument_id=10, close=110.0)
        self.log.settle(current_cycle=2)
        text = self.path.read_text(encoding="utf-8").strip()
        self.assertNotEqual(text, "")
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        self.assertEqual(rows[0]["tool"], "sma_cross")
        self.assertTrue(rows[0]["hit"])

    def test_unknown_tool_returns_zero_stats(self) -> None:
        stats = self.log.stats_for("never_seen")
        self.assertEqual(stats.observations, 0)
        self.assertEqual(stats.hit_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
