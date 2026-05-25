"""Tests for NewsScheduler interval gating."""

import unittest
from typing import Any
from unittest.mock import MagicMock

from src.news.aggregator import AggregatorRunStats, NewsAggregator
from src.news.scheduler import NewsScheduler


def _stub_aggregator(stats: AggregatorRunStats | None = None) -> NewsAggregator:
    """Return a MagicMock conforming to the NewsAggregator surface used here."""
    m = MagicMock(spec=NewsAggregator)
    m.run.return_value = stats or AggregatorRunStats(
        started_at_unix=0.0, finished_at_unix=0.0,
    )
    return m


class NewsSchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.t = [0.0]  # mutable clock
        self.clock = lambda: self.t[0]

    def test_first_call_runs_immediately(self) -> None:
        agg = _stub_aggregator()
        sched = NewsScheduler(agg, interval_minutes=60, clock=self.clock)
        stats = sched.maybe_run()
        self.assertIsNotNone(stats)
        agg.run.assert_called_once()

    def test_second_call_within_interval_skips(self) -> None:
        agg = _stub_aggregator()
        sched = NewsScheduler(agg, interval_minutes=60, clock=self.clock)
        sched.maybe_run()
        self.t[0] += 30 * 60  # 30 min later — still inside 60-min interval
        result = sched.maybe_run()
        self.assertIsNone(result)
        self.assertEqual(agg.run.call_count, 1)

    def test_runs_again_after_interval(self) -> None:
        agg = _stub_aggregator()
        sched = NewsScheduler(agg, interval_minutes=60, clock=self.clock)
        sched.maybe_run()
        self.t[0] += 60 * 60 + 1
        sched.maybe_run()
        self.assertEqual(agg.run.call_count, 2)

    def test_force_overrides_interval(self) -> None:
        agg = _stub_aggregator()
        sched = NewsScheduler(agg, interval_minutes=60, clock=self.clock)
        sched.maybe_run()
        self.t[0] += 10  # well inside interval
        sched.maybe_run(force=True)
        self.assertEqual(agg.run.call_count, 2)

    def test_known_symbols_passed_through(self) -> None:
        agg = _stub_aggregator()
        sched = NewsScheduler(agg, interval_minutes=60, clock=self.clock)
        sched.maybe_run(known_symbols=["AAPL", "MSFT"])
        agg.run.assert_called_once_with(known_symbols=["AAPL", "MSFT"])

    def test_aggregator_failure_does_not_crash(self) -> None:
        agg = _stub_aggregator()
        agg.run.side_effect = RuntimeError("boom")
        sched = NewsScheduler(agg, interval_minutes=60, clock=self.clock)
        result = sched.maybe_run()
        self.assertIsNone(result)
        # Failure does not advance the clock — next call should also try.
        agg.run.side_effect = None
        result2 = sched.maybe_run()
        self.assertIsNotNone(result2)

    def test_last_stats_exposed(self) -> None:
        stats = AggregatorRunStats(items_kept=7)
        agg = _stub_aggregator(stats=stats)
        sched = NewsScheduler(agg, interval_minutes=60, clock=self.clock)
        sched.maybe_run()
        self.assertIs(sched.last_stats, stats)

    def test_seconds_until_next_run(self) -> None:
        agg = _stub_aggregator()
        sched = NewsScheduler(agg, interval_minutes=60, clock=self.clock)
        self.assertEqual(sched.seconds_until_next_run(), 0.0)
        sched.maybe_run()
        self.t[0] += 30 * 60
        self.assertAlmostEqual(sched.seconds_until_next_run(), 30 * 60, places=2)


if __name__ == "__main__":
    unittest.main()
