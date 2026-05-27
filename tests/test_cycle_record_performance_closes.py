"""Regression test: bot-initiated CLOSE must reach PerformanceTracker.

When the bot itself closes a position (LLM-driven, synthetic SL/TP,
directive-driven), the executor removes the pid from
``state.bot_owned_positions`` and the broker drops it from the next
portfolio snapshot. The legacy ``_record_performance_vanished`` fallback
only handles positions still in ``state.bot_owned_positions``, so
without this hook every bot-initiated close leaked in the tracker's
``_open`` set forever and never landed in ``closed_trades``.

User-visible symptom (the bug we're guarding against): /stats showed
"Open bot positions: 9", DAILY showed "9 trades", but OVERVIEW kept
saying "Trades today: 0" / "$0 realized" / "No closed bot trades yet"
because :meth:`PerformanceTracker.record_close` was never called.
"""

from __future__ import annotations

import logging
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.cycle import CycleRunner
from src.execution.executor import ExecutionResult
from src.performance import PerformanceTracker
from src.state import BotState


class _CycleRunnerStub(CycleRunner):
    """Bypass __init__ — we only test the close-booking hook in isolation."""

    def __init__(  # noqa: D401 - test plumbing
        self,
        *,
        performance: PerformanceTracker | None,
        state: BotState,
        logger: logging.Logger,
    ) -> None:
        # We deliberately skip CycleRunner.__init__ because we only
        # need the slice that books closes. Setting only the attributes
        # the method touches keeps the test fast + isolated.
        self._performance = performance
        self._state = state
        self._log = logger


def _make_tracker(tmpdir: Path) -> PerformanceTracker:
    return PerformanceTracker(tmpdir)


def _open(
    tracker: PerformanceTracker,
    *,
    pid: int,
    symbol: str = "AMZN",
    amount: float = 350.0,
) -> None:
    tracker.record_open(
        position_id=pid,
        instrument_id=1000 + pid,
        symbol=symbol,
        asset_class="stock",
        is_buy=True,
        amount_usd=amount,
        units=1.0,
        open_rate=100.0,
        opened_at=datetime(2026, 5, 27, 16, 39, tzinfo=timezone.utc),
    )


def _close_result(
    *,
    pid: int,
    symbol: str = "AMZN",
    status: str = "ok",
    detail: str = "orderID=999000",
) -> ExecutionResult:
    return ExecutionResult(
        request_symbol=symbol,
        action="CLOSE",
        status=status,
        order_id=999000,
        position_id=pid,
        instrument_id=1000 + pid,
        detail=detail,
    )


class BotInitiatedCloseBookingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmpdir = Path(self._tmp.name)
        self.tracker = _make_tracker(self.tmpdir)
        self.state = BotState()
        log = logging.getLogger("test.cycle.perf_closes")
        log.addHandler(logging.NullHandler())
        self.runner = _CycleRunnerStub(
            performance=self.tracker, state=self.state, logger=log,
        )

    def test_full_close_books_realized_trade(self) -> None:
        _open(self.tracker, pid=1001)
        # Executor would have removed pid from state on a full close.
        # We mirror that to drive the deterministic partial-vs-full
        # detection in ``_record_performance_closes``.
        self.assertNotIn(1001, self.state.bot_owned_positions)
        self.assertEqual(len(self.tracker.open_positions()), 1)

        self.runner._record_performance_closes([_close_result(pid=1001)])

        self.assertEqual(len(self.tracker.open_positions()), 0)
        closed = self.tracker.closed_trades()
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0].position_id, 1001)
        self.assertEqual(closed[0].close_reason, "bot")

    def test_multiple_closes_in_one_cycle_all_booked(self) -> None:
        for pid in (2001, 2002, 2003):
            _open(self.tracker, pid=pid, symbol=f"SYM{pid}")
        self.runner._record_performance_closes([
            _close_result(pid=2001, symbol="SYM2001"),
            _close_result(pid=2002, symbol="SYM2002"),
            _close_result(pid=2003, symbol="SYM2003"),
        ])
        self.assertEqual(len(self.tracker.open_positions()), 0)
        self.assertEqual(len(self.tracker.closed_trades()), 3)

    def test_partial_close_keeps_position_open(self) -> None:
        # On a partial close, the executor leaves the pid in
        # bot_owned_positions because the broker still has the remainder.
        _open(self.tracker, pid=3001)
        self.state.add_owned(3001)
        self.runner._record_performance_closes([
            _close_result(
                pid=3001,
                detail="orderID=999000, units_to_deduct=0.500000",
            ),
        ])
        # The position must STAY in _open — it isn't fully closed yet.
        self.assertEqual(len(self.tracker.open_positions()), 1)
        self.assertEqual(len(self.tracker.closed_trades()), 0)

    def test_failed_close_does_not_book(self) -> None:
        _open(self.tracker, pid=4001)
        # Even though the pid isn't in state (e.g. fresh restart), a
        # non-ok status must never produce a realized trade.
        self.runner._record_performance_closes([
            _close_result(pid=4001, status="failed"),
            _close_result(pid=4001, status="ambiguous"),
            _close_result(pid=4001, status="rate_limited"),
            _close_result(pid=4001, status="skipped"),
        ])
        self.assertEqual(len(self.tracker.open_positions()), 1)
        self.assertEqual(len(self.tracker.closed_trades()), 0)

    def test_buy_result_is_ignored(self) -> None:
        _open(self.tracker, pid=5001)
        buy_result = ExecutionResult(
            request_symbol="AMD", action="BUY", status="ok",
            order_id=1234, instrument_id=42, amount_usd=300.0,
            detail="orderID=1234",
        )
        self.runner._record_performance_closes([buy_result])
        self.assertEqual(len(self.tracker.open_positions()), 1)
        self.assertEqual(len(self.tracker.closed_trades()), 0)

    def test_close_for_unknown_position_is_a_noop(self) -> None:
        # Closing a pid the tracker has never seen (e.g. a stale
        # ExecutionResult from before the tracker existed) must not
        # crash and must not log a phantom trade.
        self.runner._record_performance_closes([_close_result(pid=9999)])
        self.assertEqual(len(self.tracker.closed_trades()), 0)

    def test_no_tracker_wired_is_safe(self) -> None:
        runner = _CycleRunnerStub(
            performance=None,
            state=self.state,
            logger=logging.getLogger("test.cycle.no_tracker"),
        )
        runner._record_performance_closes([_close_result(pid=6001)])

    def test_summary_reflects_booked_close(self) -> None:
        # End-to-end shape: the user's /stats sees the realized trade
        # appear in by_period / by_symbol / closed_trades.
        _open(self.tracker, pid=7001, symbol="AMZN", amount=350.0)
        self.runner._record_performance_closes([
            _close_result(pid=7001, symbol="AMZN"),
        ])
        summary = self.tracker.summary()
        self.assertEqual(summary["bot"]["open_position_count"], 0)
        self.assertEqual(summary["bot"]["trades_total"], 1)
        all_window = summary["by_period"]["all"]
        self.assertEqual(all_window["trades"], 1)
        self.assertEqual(
            [r["symbol"] for r in self.tracker.by_symbol()], ["AMZN"],
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
