"""Tests for src.execution.session — pure session-boundary math."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from src.execution.session import (
    equity_session_window,
    is_market_open,
    seconds_since_open,
    session_state,
)
from src.strategy.tools.base import AssetClass


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


class CryptoSessionTests(unittest.TestCase):
    """Crypto is always open — no session boundaries apply."""

    def test_always_open(self) -> None:
        for now in (
            _utc(2026, 5, 25, 3, 0),    # Monday 3 AM UTC
            _utc(2026, 5, 24, 13, 0),   # Sunday 1 PM UTC
            _utc(2026, 5, 26, 23, 59),  # Tuesday 11:59 PM UTC
        ):
            self.assertTrue(is_market_open(AssetClass.CRYPTO, now))

    def test_seconds_since_open_is_none(self) -> None:
        """Crypto reports None so callers fall back to absolute placement age."""
        self.assertIsNone(
            seconds_since_open(AssetClass.CRYPTO, _utc(2026, 5, 26, 15)),
        )


class FxSessionTests(unittest.TestCase):
    """FX runs 24x5 — closed Sat/Sun."""

    def test_open_on_weekday(self) -> None:
        for hour in (0, 6, 13, 21, 23):
            self.assertTrue(is_market_open(AssetClass.FX, _utc(2026, 5, 26, hour)))

    def test_closed_on_weekend(self) -> None:
        # 2026-05-23 is Saturday, 2026-05-24 is Sunday.
        self.assertFalse(is_market_open(AssetClass.FX, _utc(2026, 5, 23, 10)))
        self.assertFalse(is_market_open(AssetClass.FX, _utc(2026, 5, 24, 10)))

    def test_weekend_state_reports_next_open_monday(self) -> None:
        state = session_state(AssetClass.FX, _utc(2026, 5, 23, 10))
        self.assertFalse(state.is_open)
        self.assertIsNotNone(state.next_open_utc)
        assert state.next_open_utc is not None  # for mypy
        self.assertEqual(state.next_open_utc.weekday(), 0)
        self.assertEqual(state.next_open_utc.hour, 0)


class EquitySessionTests(unittest.TestCase):
    """US-equity session: 09:30–16:00 New York local time, DST-aware."""

    # ---- EDT (summer) — 13:30–20:00 UTC -----------------------------

    def test_edt_open_inside_session(self) -> None:
        # 2026-05-26 is a Tuesday in EDT (NY summer time).
        self.assertTrue(
            is_market_open(AssetClass.STOCK, _utc(2026, 5, 26, 15, 0)),
        )
        # 19:59 UTC = 15:59 EDT, last minute of session.
        self.assertTrue(
            is_market_open(AssetClass.STOCK, _utc(2026, 5, 26, 19, 59)),
        )

    def test_edt_pre_open_closed(self) -> None:
        # 8 AM UTC = 4 AM EDT, well before 09:30 EDT (13:30 UTC) open.
        self.assertFalse(
            is_market_open(AssetClass.STOCK, _utc(2026, 5, 26, 8, 0)),
        )

    def test_edt_post_close_closed(self) -> None:
        # 20:00 UTC = 16:00 EDT — exact close, treated as already closed.
        self.assertFalse(
            is_market_open(AssetClass.STOCK, _utc(2026, 5, 26, 20, 0)),
        )
        # 20:30 UTC is 30 min after the EDT bell. THE bug the user hit:
        # before the DST fix, this returned True because the hard-coded
        # window said 21:00 UTC close.
        self.assertFalse(
            is_market_open(AssetClass.STOCK, _utc(2026, 5, 26, 20, 30)),
        )

    # ---- EST (winter) — 14:30–21:00 UTC -----------------------------

    def test_est_open_inside_session(self) -> None:
        # 2026-01-13 is a Tuesday in EST (NY winter time).
        # 15:00 UTC = 10:00 EST → mid-session.
        self.assertTrue(
            is_market_open(AssetClass.STOCK, _utc(2026, 1, 13, 15, 0)),
        )
        # 20:59 UTC = 15:59 EST, last minute of session.
        self.assertTrue(
            is_market_open(AssetClass.STOCK, _utc(2026, 1, 13, 20, 59)),
        )

    def test_est_pre_open_closed(self) -> None:
        # 14:00 UTC = 09:00 EST — 30 min before open.
        self.assertFalse(
            is_market_open(AssetClass.STOCK, _utc(2026, 1, 13, 14, 0)),
        )

    def test_est_post_close_closed(self) -> None:
        # 21:00 UTC = 16:00 EST — exact close.
        self.assertFalse(
            is_market_open(AssetClass.STOCK, _utc(2026, 1, 13, 21, 0)),
        )

    # ---- Window helper -----------------------------------------------

    def test_equity_session_window_edt(self) -> None:
        open_utc, close_utc = equity_session_window(_utc(2026, 5, 26, 12, 0))
        self.assertEqual((open_utc.hour, open_utc.minute), (13, 30))
        self.assertEqual((close_utc.hour, close_utc.minute), (20, 0))

    def test_equity_session_window_est(self) -> None:
        open_utc, close_utc = equity_session_window(_utc(2026, 1, 13, 12, 0))
        self.assertEqual((open_utc.hour, open_utc.minute), (14, 30))
        self.assertEqual((close_utc.hour, close_utc.minute), (21, 0))

    # ---- General invariants -----------------------------------------

    def test_weekend_closed(self) -> None:
        # Saturday at 15:00 UTC — inside the EDT time window but wrong day.
        self.assertFalse(
            is_market_open(AssetClass.STOCK, _utc(2026, 5, 23, 15, 0)),
        )

    def test_seconds_since_open_at_boundary_edt(self) -> None:
        # Exactly at EDT open → 0 seconds.
        self.assertEqual(
            seconds_since_open(AssetClass.STOCK, _utc(2026, 5, 26, 13, 30)),
            0,
        )
        # 30 minutes after EDT open.
        self.assertEqual(
            seconds_since_open(AssetClass.STOCK, _utc(2026, 5, 26, 14, 0)),
            30 * 60,
        )

    def test_seconds_since_open_when_closed(self) -> None:
        self.assertIsNone(
            seconds_since_open(AssetClass.STOCK, _utc(2026, 5, 26, 8, 0)),
        )

    def test_naive_datetime_is_assumed_utc(self) -> None:
        """Tolerate a naive datetime by treating it as UTC."""
        naive = datetime(2026, 5, 26, 15, 0)
        self.assertTrue(is_market_open(AssetClass.STOCK, naive))

    def test_next_open_is_strictly_future_edt(self) -> None:
        # Friday after EDT close → next open is Monday 13:30 UTC.
        state = session_state(AssetClass.STOCK, _utc(2026, 5, 22, 22, 0))
        assert state.next_open_utc is not None
        self.assertEqual(state.next_open_utc.weekday(), 0)
        self.assertEqual(state.next_open_utc.hour, 13)
        self.assertEqual(state.next_open_utc.minute, 30)

    def test_next_open_is_strictly_future_est(self) -> None:
        # Friday after EST close → next open is Monday 14:30 UTC.
        state = session_state(AssetClass.STOCK, _utc(2026, 1, 9, 22, 0))
        assert state.next_open_utc is not None
        self.assertEqual(state.next_open_utc.weekday(), 0)
        self.assertEqual(state.next_open_utc.hour, 14)
        self.assertEqual(state.next_open_utc.minute, 30)


if __name__ == "__main__":
    unittest.main()
