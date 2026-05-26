"""Tests for :mod:`src.performance`.

We exercise the three public hooks (open / observe / close), the
summary aggregation, the day-rollover behaviour, and persistence
across a fresh instantiation (simulating bot restart).
"""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from src.performance import PerformanceTracker
from src.performance.aggregations import aggregate, by_symbol, filter_window
from src.performance.types import RealizedTrade


class _FakeBrokerPosition:
    """Mimics the public attributes of :class:`src.etoro.trading.Position`."""

    def __init__(self, *, position_id: int, pnl: float, mark: float = 0.0) -> None:
        self.position_id = position_id
        self.pnl = pnl
        self.mark = mark


# ----------------------------------------------------------------------
# Open / observe / close happy path
# ----------------------------------------------------------------------

class OpenObserveCloseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = TemporaryDirectory()
        self.now = datetime(2026, 5, 26, 13, 0, 0, tzinfo=timezone.utc)
        self.tracker = PerformanceTracker(
            Path(self.tmpdir.name),
            now_fn=self._clock,
        )

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _clock(self) -> datetime:
        return self.now

    def _advance(self, **delta) -> None:
        self.now = self.now + timedelta(**delta)

    def test_record_open_persists_to_disk(self) -> None:
        self.tracker.record_open(
            position_id=9001, instrument_id=1832, symbol="AMD",
            asset_class="stock", is_buy=True,
            amount_usd=500.0, units=1.0, open_rate=500.0,
        )
        open_positions = self.tracker.open_positions()
        self.assertEqual(len(open_positions), 1)
        self.assertEqual(open_positions[0].symbol, "AMD")

        # Restart simulation: a fresh tracker hydrates from the same dir.
        revived = PerformanceTracker(Path(self.tmpdir.name), now_fn=self._clock)
        self.assertEqual(len(revived.open_positions()), 1)
        self.assertEqual(revived.open_positions()[0].position_id, 9001)

    def test_observe_updates_mfe_mae(self) -> None:
        self.tracker.record_open(
            position_id=9001, instrument_id=1, symbol="AAA",
            asset_class="stock", is_buy=True,
            amount_usd=100.0, units=1.0, open_rate=100.0,
        )
        # First observation: position is up +5
        self.tracker.observe_positions(
            bot_positions=[_FakeBrokerPosition(position_id=9001, pnl=5.0)],
            equity=1000.0, account_unrealized_pnl=5.0,
        )
        # Second: position drops to -8
        self._advance(minutes=5)
        self.tracker.observe_positions(
            bot_positions=[_FakeBrokerPosition(position_id=9001, pnl=-8.0)],
            equity=987.0, account_unrealized_pnl=-8.0,
        )
        # Third: recovers to +3
        self._advance(minutes=5)
        self.tracker.observe_positions(
            bot_positions=[_FakeBrokerPosition(position_id=9001, pnl=3.0)],
            equity=998.0, account_unrealized_pnl=3.0,
        )
        s = self.tracker.open_positions()[0]
        self.assertAlmostEqual(s.mfe_usd, 5.0)
        self.assertAlmostEqual(s.mae_usd, -8.0)
        self.assertAlmostEqual(s.last_pnl_usd, 3.0)
        self.assertEqual(s.snapshots, 3)

    def test_record_close_creates_realized_trade(self) -> None:
        self.tracker.record_open(
            position_id=9001, instrument_id=1, symbol="AAA",
            asset_class="stock", is_buy=True,
            amount_usd=100.0, units=1.0, open_rate=100.0,
        )
        self.tracker.observe_positions(
            bot_positions=[_FakeBrokerPosition(position_id=9001, pnl=4.0)],
            equity=1004.0, account_unrealized_pnl=4.0,
        )
        self._advance(hours=2)
        trade = self.tracker.record_close(
            position_id=9001, close_rate=104.0, realized_pnl_usd=4.0,
        )
        self.assertIsNotNone(trade)
        self.assertEqual(trade.symbol, "AAA")
        self.assertAlmostEqual(trade.realized_pnl_usd, 4.0)
        self.assertAlmostEqual(trade.realized_pnl_pct, 4.0)
        self.assertEqual(trade.hold_seconds, 2 * 3600)
        # The open ledger should be empty.
        self.assertEqual(self.tracker.open_positions(), [])
        # Closed ledger contains the trade.
        closed = self.tracker.closed_trades()
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0].position_id, 9001)

    def test_close_falls_back_to_last_observed_pnl(self) -> None:
        """If the caller doesn't pass realized_pnl_usd, use last observed."""
        self.tracker.record_open(
            position_id=9001, instrument_id=1, symbol="AAA",
            asset_class="stock", is_buy=True,
            amount_usd=200.0, units=2.0, open_rate=100.0,
        )
        self.tracker.observe_positions(
            bot_positions=[_FakeBrokerPosition(position_id=9001, pnl=-12.0)],
            equity=988.0, account_unrealized_pnl=-12.0,
        )
        trade = self.tracker.record_close(position_id=9001)
        self.assertIsNotNone(trade)
        self.assertAlmostEqual(trade.realized_pnl_usd, -12.0)
        self.assertAlmostEqual(trade.realized_pnl_pct, -6.0)

    def test_close_of_unknown_position_returns_none(self) -> None:
        self.assertIsNone(self.tracker.record_close(position_id=42))


# ----------------------------------------------------------------------
# Summary windowing
# ----------------------------------------------------------------------

class SummaryWindowingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = TemporaryDirectory()
        # Pin time so the windowing assertions don't drift.
        self.now = datetime(2026, 5, 26, 23, 30, 0, tzinfo=timezone.utc)
        self.tracker = PerformanceTracker(
            Path(self.tmpdir.name), now_fn=lambda: self.now,
        )

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _make_closed(self, *, pid: int, pnl: float, closed_at: datetime) -> None:
        # Use the tracker's hooks so the open/close pair flows through
        # the daily roller too.
        opening = self.now
        self.now = closed_at - timedelta(hours=1)
        self.tracker.record_open(
            position_id=pid, instrument_id=pid, symbol=f"S{pid}",
            asset_class="stock", is_buy=True,
            amount_usd=100.0, units=1.0, open_rate=100.0,
        )
        self.tracker.observe_positions(
            bot_positions=[_FakeBrokerPosition(position_id=pid, pnl=pnl)],
            equity=1000.0 + pnl, account_unrealized_pnl=pnl,
        )
        self.now = closed_at
        self.tracker.record_close(position_id=pid, realized_pnl_usd=pnl)
        self.now = opening

    def test_summary_partitions_into_today_7d_30d_all(self) -> None:
        # Trade 1 today (+10), trade 2 yesterday (-5), trade 3 10 days ago (+20),
        # trade 4 100 days ago (-15).
        self._make_closed(pid=1, pnl=10.0, closed_at=self.now - timedelta(hours=2))
        self._make_closed(pid=2, pnl=-5.0, closed_at=self.now - timedelta(days=1, hours=2))
        self._make_closed(pid=3, pnl=20.0, closed_at=self.now - timedelta(days=10))
        self._make_closed(pid=4, pnl=-15.0, closed_at=self.now - timedelta(days=100))

        summary = self.tracker.summary()
        # Today: only trade 1.
        self.assertEqual(summary["by_period"]["today"]["trades"], 1)
        self.assertAlmostEqual(summary["by_period"]["today"]["realized_pnl_usd"], 10.0)
        # 7d: trade 1 + 2.
        self.assertEqual(summary["by_period"]["7d"]["trades"], 2)
        self.assertAlmostEqual(summary["by_period"]["7d"]["realized_pnl_usd"], 5.0)
        # 30d: trades 1 + 2 + 3.
        self.assertEqual(summary["by_period"]["30d"]["trades"], 3)
        self.assertAlmostEqual(summary["by_period"]["30d"]["realized_pnl_usd"], 25.0)
        # All-time: every trade.
        self.assertEqual(summary["by_period"]["all"]["trades"], 4)
        self.assertAlmostEqual(summary["by_period"]["all"]["realized_pnl_usd"], 10.0)

    def test_by_symbol_aggregation(self) -> None:
        self._make_closed(pid=1, pnl=10.0, closed_at=self.now - timedelta(hours=1))
        self._make_closed(pid=2, pnl=-3.0, closed_at=self.now - timedelta(hours=2))
        self._make_closed(pid=3, pnl=5.0, closed_at=self.now - timedelta(hours=3))
        # Three distinct symbols ("S1", "S2", "S3").
        rows = self.tracker.by_symbol()
        self.assertEqual(len(rows), 3)
        # Sorted by realized_pnl descending.
        self.assertEqual(rows[0]["symbol"], "S1")
        self.assertEqual(rows[-1]["symbol"], "S2")


# ----------------------------------------------------------------------
# Pure aggregation helpers
# ----------------------------------------------------------------------

class AggregationHelperTests(unittest.TestCase):
    def test_aggregate_empty(self) -> None:
        result = aggregate([])
        self.assertEqual(result["trades"], 0)
        self.assertEqual(result["win_rate_pct"], 0.0)

    def test_aggregate_winrate(self) -> None:
        trades = [
            _trade(pnl=10.0), _trade(pnl=5.0),
            _trade(pnl=-3.0), _trade(pnl=0.0),
        ]
        r = aggregate(trades)
        self.assertEqual(r["trades"], 4)
        self.assertEqual(r["wins"], 2)
        self.assertEqual(r["losses"], 1)
        self.assertEqual(r["breakeven"], 1)
        # Win rate counts only true wins (50% here).
        self.assertEqual(r["win_rate_pct"], 50.0)
        self.assertAlmostEqual(r["realized_pnl_usd"], 12.0)
        self.assertAlmostEqual(r["avg_win_usd"], 7.5)
        self.assertAlmostEqual(r["avg_loss_usd"], -3.0)
        self.assertAlmostEqual(r["biggest_win_usd"], 10.0)
        self.assertAlmostEqual(r["biggest_loss_usd"], -3.0)


def _trade(*, pnl: float, hold: int = 3600) -> RealizedTrade:
    return RealizedTrade(
        position_id=1, instrument_id=1, symbol="X", asset_class="stock",
        is_buy=True, amount_usd=100.0, units=1.0, open_rate=100.0,
        close_rate=100.0 + pnl, realized_pnl_usd=pnl, realized_pnl_pct=pnl,
        opened_at_iso="2026-05-26T10:00:00Z",
        closed_at_iso="2026-05-26T11:00:00Z",
        hold_seconds=hold, mfe_usd=max(0.0, pnl), mae_usd=min(0.0, pnl),
        close_reason="reconciled",
    )


if __name__ == "__main__":
    unittest.main()
