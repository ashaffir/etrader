"""Performance tracker — captures bot trade outcomes over time.

Three hooks the cycle calls each cycle:

1. :meth:`PerformanceTracker.record_open` — on a successful BUY,
   register the entry (price, units, amount, asset class).
2. :meth:`PerformanceTracker.observe_positions` — once per cycle with
   the live bot-owned positions; updates mark-to-market and MFE/MAE
   for each and rolls the day's equity high/low into the daily snapshot.
3. :meth:`PerformanceTracker.record_close` — when reconcile detects
   that a tracked position has vanished from the broker, compute the
   realized P/L from the last observed mark and append a
   :class:`RealizedTrade` to the ledger.

:meth:`summary` returns the structured payload used by ``/stats``
and the ``/ask`` LLM.

Thread-safety: every public method holds an internal lock so the
cycle loop and the Telegram /stats path can call concurrently.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .aggregations import aggregate, by_symbol, filter_window
from .daily_roller import DailyRoller
from .storage import PerformanceStorage
from .types import DailySnapshot, OpenTradeState, RealizedTrade


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


class PerformanceTracker:
    def __init__(
        self,
        data_dir: Path,
        *,
        logger: logging.Logger | None = None,
        now_fn=_utcnow,
    ) -> None:
        self._storage = PerformanceStorage(data_dir, logger=logger)
        self._logger = logger or logging.getLogger("etrader.performance")
        self._now = now_fn
        self._lock = threading.RLock()
        self._open: dict[int, OpenTradeState] = self._storage.load_open_positions()
        self._daily = DailyRoller(self._storage, logger=self._logger)

    # ------------------------------------------------------------------
    # Hooks called from the cycle
    # ------------------------------------------------------------------

    def record_open(
        self,
        *,
        position_id: int,
        instrument_id: int,
        symbol: str,
        asset_class: str,
        is_buy: bool,
        amount_usd: float,
        units: float,
        open_rate: float,
        opened_at: datetime | None = None,
    ) -> None:
        """Register a newly-opened bot position."""
        if position_id <= 0:
            return
        opened = opened_at or self._now()
        state = OpenTradeState(
            position_id=int(position_id),
            instrument_id=int(instrument_id),
            symbol=str(symbol or "").upper(),
            asset_class=str(asset_class or "other"),
            is_buy=bool(is_buy),
            amount_usd=float(amount_usd or 0.0),
            units=float(units or 0.0),
            open_rate=float(open_rate or 0.0),
            opened_at_iso=_iso(opened),
        )
        with self._lock:
            self._open[state.position_id] = state
            self._daily.bump_open(now=self._now())
            self._storage.save_open_positions(self._open)

    def observe_positions(
        self,
        *,
        bot_positions: Iterable[Any],
        equity: float | None,
        account_unrealized_pnl: float | None,
    ) -> None:
        """Mark-to-market every still-open bot trade and roll the day."""
        ts = self._now()
        with self._lock:
            for p in bot_positions:
                pid = int(getattr(p, "position_id", 0) or 0)
                state = self._open.get(pid)
                if state is None:
                    continue
                _apply_mark(state, p, ts)
            self._storage.save_open_positions(self._open)
            self._daily.update_equity(
                now=ts,
                equity=equity,
                account_unrealized=account_unrealized_pnl,
                bot_unrealized=sum(
                    float(p.last_pnl_usd or 0.0) for p in self._open.values()
                ),
            )

    def record_close(
        self,
        *,
        position_id: int,
        close_rate: float | None = None,
        realized_pnl_usd: float | None = None,
        closed_at: datetime | None = None,
        reason: str = "reconciled",
    ) -> RealizedTrade | None:
        """Move a tracked position from open → closed ledger."""
        with self._lock:
            state = self._open.pop(int(position_id), None)
            if state is None:
                return None
            closed = closed_at or self._now()
            pnl = (
                float(realized_pnl_usd)
                if realized_pnl_usd is not None
                else float(state.last_pnl_usd or 0.0)
            )
            pnl_pct = (
                (pnl / state.amount_usd) * 100.0
                if state.amount_usd > 0 else 0.0
            )
            try:
                opened = datetime.fromisoformat(state.opened_at_iso.replace("Z", "+00:00"))
            except ValueError:
                opened = closed
            hold_seconds = max(0, int((closed - opened).total_seconds()))
            trade = RealizedTrade(
                position_id=state.position_id,
                instrument_id=state.instrument_id,
                symbol=state.symbol,
                asset_class=state.asset_class,
                is_buy=state.is_buy,
                amount_usd=state.amount_usd,
                units=state.units,
                open_rate=state.open_rate,
                close_rate=close_rate if close_rate is not None else state.last_mark,
                realized_pnl_usd=pnl,
                realized_pnl_pct=pnl_pct,
                opened_at_iso=state.opened_at_iso,
                closed_at_iso=_iso(closed),
                hold_seconds=hold_seconds,
                mfe_usd=state.mfe_usd,
                mae_usd=state.mae_usd,
                close_reason=str(reason or "reconciled"),
            )
            self._storage.append_closed_trade(trade)
            self._storage.save_open_positions(self._open)
            self._daily.bump_close(now=closed, trade=trade)
            return trade

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def open_positions(self) -> list[OpenTradeState]:
        with self._lock:
            return list(self._open.values())

    def open_states_by_position(self) -> dict[int, OpenTradeState]:
        """Indexed view used by the position reviewer and decision prompt."""
        with self._lock:
            return dict(self._open)

    def closed_trades(self, *, limit: int | None = None) -> list[RealizedTrade]:
        return self._storage.read_closed_trades(limit=limit)

    def daily_history(self, *, limit: int | None = None) -> list[DailySnapshot]:
        with self._lock:
            self._daily.flush_if_dirty()
        return self._storage.read_dailies(limit=limit)

    def by_symbol(self) -> list[dict[str, Any]]:
        return by_symbol(self._storage.read_closed_trades())

    def summary(
        self,
        *,
        periods: tuple[str, ...] = ("today", "7d", "30d", "all"),
    ) -> dict[str, Any]:
        """Return the structured payload used by /stats and the LLM."""
        with self._lock:
            self._daily.flush_if_dirty()
            open_positions = list(self._open.values())
            account_pnl = self._daily.account_unrealized()
        closed = self._storage.read_closed_trades()
        now = self._now()
        per_window = {
            period: aggregate(filter_window(closed, now=now, period=period))
            for period in periods
        }
        bot_unrealized = sum(
            float(p.last_pnl_usd or 0.0) for p in open_positions
        )
        realized_total = sum(t.realized_pnl_usd for t in closed)
        realized_today = per_window.get("today", {}).get("realized_pnl_usd", 0.0)
        return {
            "bot": {
                "unrealized_pnl_usd": round(bot_unrealized, 2),
                "open_position_count": len(open_positions),
                "realized_pnl_total_usd": round(realized_total, 2),
                "realized_pnl_today_usd": round(realized_today, 2),
                "trades_total": len(closed),
            },
            "account": {
                "unrealized_pnl_usd": (
                    round(account_pnl, 2) if account_pnl is not None else None
                ),
            },
            "open": [p.to_dict() for p in open_positions],
            "by_period": per_window,
        }


# ----------------------------------------------------------------------
# Internals — applied per-position on observe_positions
# ----------------------------------------------------------------------

def _apply_mark(state: OpenTradeState, p: Any, ts: datetime) -> None:
    """Update ``state`` from a live broker position object ``p``."""
    pnl_usd = float(getattr(p, "pnl", 0.0) or 0.0)
    mark = float(getattr(p, "mark", 0.0) or 0.0)
    if mark <= 0 and state.amount_usd > 0:
        # Synthetic mark derived from open rate + relative P/L.
        delta = pnl_usd / state.amount_usd
        if not state.is_buy:
            delta = -delta
        mark = state.open_rate * (1.0 + delta)
    pct = (
        (pnl_usd / state.amount_usd) * 100.0
        if state.amount_usd > 0 else 0.0
    )
    state.last_mark = mark or state.last_mark
    state.last_pnl_usd = pnl_usd
    state.last_pnl_pct = pct
    state.last_seen_iso = _iso(ts)
    state.mfe_usd = max(state.mfe_usd, pnl_usd)
    state.mae_usd = min(state.mae_usd, pnl_usd)
    state.snapshots += 1
