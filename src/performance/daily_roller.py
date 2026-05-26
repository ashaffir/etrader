"""Daily-snapshot rolling logic — separated from :class:`PerformanceTracker`
to keep that class focused on per-trade events.

A ``DailyRoller`` owns the in-memory :class:`DailySnapshot` for *today*
and knows how to:

- start a new row on the first call of a new UTC day,
- flush the previous day's row to the JSONL ledger on rollover,
- update the high/low/close fields each cycle,
- bump the per-day counters when trades open or close,
- rewrite the most-recent line in the ledger so an in-progress day
  is always reflected on disk.

All methods assume the caller holds the tracker's lock.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

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
        """Persist today's running snapshot to JSONL."""
        if self._today is None or not self._dirty:
            return
        existing = self._storage.read_dailies()
        if existing and existing[-1].date_iso == self._today.date_iso:
            self._rewrite(existing[:-1] + [self._today])
        else:
            self._storage.append_daily(self._today)
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
        if self._today is not None and self._dirty:
            self._storage.append_daily(self._today)
        self._today = DailySnapshot(date_iso=today_iso)
        self._dirty = False
        return self._today

    def _rewrite(self, rows: Iterable[DailySnapshot]) -> None:
        path = self._storage._daily_path  # noqa: SLF001
        try:
            import json
            with path.open("w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r.to_dict(), default=str) + "\n")
        except OSError as exc:
            self._logger.warning("perf daily rewrite failed: %s", exc)
