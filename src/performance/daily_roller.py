"""Daily-snapshot rolling logic — separated from :class:`PerformanceTracker`
to keep that class focused on per-trade events.

A ``DailyRoller`` owns the in-memory :class:`DailySnapshot` for *today*
and knows how to:

- start a new row on the first call of a new UTC day,
- finalize the previous day's row in storage on rollover,
- update the high/low/close fields each cycle,
- bump the per-day counters when trades open or close,
- upsert the running day's row so the on-disk snapshot is always
  in sync with what the bot is observing right now.

All methods assume the caller holds the tracker's lock.
"""

from __future__ import annotations

import logging
from datetime import datetime

from .storage import PerformanceStorage
from .types import DailySnapshot, RealizedTrade


class DailyRoller:
    def __init__(
        self,
        storage: PerformanceStorage,
        *,
        logger: logging.Logger,
    ) -> None:
        self._storage = storage
        self._logger = logger
        self._today: DailySnapshot | None = None
        self._dirty = False

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def update_equity(
        self,
        *,
        now: datetime,
        equity: float | None,
        account_unrealized: float | None,
        bot_unrealized: float,
    ) -> None:
        snap = self._ensure_today(now)
        if equity is not None:
            if snap.equity_open is None:
                snap.equity_open = equity
                snap.equity_high = equity
                snap.equity_low = equity
            snap.equity_close = equity
            snap.equity_high = max(snap.equity_high or equity, equity)
            snap.equity_low = min(snap.equity_low or equity, equity)
        if account_unrealized is not None:
            snap.account_unrealized_close_usd = account_unrealized
        if snap.bot_unrealized_open_usd is None:
            snap.bot_unrealized_open_usd = bot_unrealized
        snap.bot_unrealized_close_usd = bot_unrealized
        self._dirty = True

    def bump_open(self, *, now: datetime) -> None:
        snap = self._ensure_today(now)
        snap.bot_trades_today += 1
        self._dirty = True

    def bump_close(self, *, now: datetime, trade: RealizedTrade) -> None:
        snap = self._ensure_today(now)
        snap.bot_realized_today_usd += trade.realized_pnl_usd
        if trade.realized_pnl_usd > 0:
            snap.bot_wins_today += 1
        elif trade.realized_pnl_usd < 0:
            snap.bot_losses_today += 1
        else:
            snap.bot_breakeven_today += 1
        self._dirty = True

    def flush_if_dirty(self) -> None:
        """Upsert today's running snapshot in the database.

        With the SQLite backend this is a single atomic upsert keyed
        on ``date_iso``, so we don't need the old "rewrite the
        ledger minus the last row" dance.
        """
        if self._today is None or not self._dirty:
            return
        self._storage.upsert_daily(self._today)
        self._dirty = False

    def today(self) -> DailySnapshot | None:
        return self._today

    def account_unrealized(self) -> float | None:
        if self._today is None:
            return None
        return self._today.account_unrealized_close_usd

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_today(self, now: datetime) -> DailySnapshot:
        today_iso = now.strftime("%Y-%m-%d")
        if self._today is not None and self._today.date_iso == today_iso:
            return self._today
        # Day boundary: flush the outgoing row before starting a new one
        # so the previous day's final snapshot is never lost.
        if self._today is not None and self._dirty:
            self._storage.upsert_daily(self._today)
        self._today = DailySnapshot(date_iso=today_iso)
        self._dirty = False
        return self._today
