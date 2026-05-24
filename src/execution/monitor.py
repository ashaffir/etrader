"""Position monitor — verifies bot-owned positions and reconciles after trades.

Two responsibilities:

1. After each cycle, re-read the portfolio (after the eToro 10s cache
   has settled — the caller decides when) and add any newly-opened
   positions to ``state.bot_owned_positions`` if their orderID matches
   one we placed this session.
2. Provide a synthetic SL/TP closer for paper backtests where the demo
   environment may not honor the SL/TP rates passed to
   ``market-open-orders/by-amount``. We compare the live rate to the
   stored SL/TP and emit close requests when triggered.

In practice eToro's demo *does* honor SL/TP, so #2 is belt-and-braces.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from ..etoro.market_data import LiveRate
from ..etoro.trading import Position, PortfolioSnapshot
from ..state import BotState


@dataclass
class TrackedOrder:
    """An order this session placed and is waiting to see fill in /pnl."""

    order_id: int
    instrument_id: int
    symbol: str
    amount_usd: float
    placed_at_monotonic: float


class PositionMonitor:
    def __init__(self, *, logger: logging.Logger | logging.LoggerAdapter | None = None) -> None:
        self._tracked: dict[int, TrackedOrder] = {}  # by order_id
        self._logger = logger or logging.getLogger("etrader.execution.monitor")

    # ------------------------------------------------------------------
    # Bot-owned-position tracking
    # ------------------------------------------------------------------

    def track_open(
        self,
        *,
        order_id: int,
        instrument_id: int,
        symbol: str,
        amount_usd: float,
        placed_at_monotonic: float,
    ) -> None:
        self._tracked[order_id] = TrackedOrder(
            order_id=order_id,
            instrument_id=instrument_id,
            symbol=symbol,
            amount_usd=amount_usd,
            placed_at_monotonic=placed_at_monotonic,
        )

    def reconcile(self, snapshot: PortfolioSnapshot, state: BotState) -> list[Position]:
        """Match tracked orders against ``snapshot.positions`` and adopt them."""
        if not self._tracked:
            return []
        tracked_order_ids = set(self._tracked.keys())
        adopted: list[Position] = []
        for pos in snapshot.positions:
            order_id = int(pos.raw.get("orderID") or pos.raw.get("orderId") or 0)
            if order_id and order_id in tracked_order_ids:
                state.add_owned(pos.position_id)
                adopted.append(pos)
                self._tracked.pop(order_id, None)
        if adopted:
            self._logger.info("[monitor] adopted %d new bot-owned position(s)", len(adopted))
        return adopted

    def expire_stale(self, *, max_age_seconds: float, now_monotonic: float) -> list[TrackedOrder]:
        stale: list[TrackedOrder] = []
        for order_id, info in list(self._tracked.items()):
            if now_monotonic - info.placed_at_monotonic >= max_age_seconds:
                stale.append(info)
                self._tracked.pop(order_id, None)
        if stale:
            self._logger.warning(
                "[monitor] %d tracked order(s) didn't materialize in %ds: %s",
                len(stale), int(max_age_seconds),
                ", ".join(f"{o.symbol}#{o.order_id}" for o in stale),
            )
        return stale

    # ------------------------------------------------------------------
    # Synthetic SL/TP (only used if the platform doesn't enforce them)
    # ------------------------------------------------------------------

    def positions_needing_close(
        self,
        *,
        bot_owned: Iterable[Position],
        rates: dict[int, LiveRate],
        stop_loss_pct: float,
        take_profit_pct: float,
    ) -> list[Position]:
        out: list[Position] = []
        for pos in bot_owned:
            rate = rates.get(pos.instrument_id)
            if rate is None or rate.mid is None or pos.open_rate <= 0:
                continue
            mid = float(rate.mid)
            change_pct = (mid - pos.open_rate) / pos.open_rate * 100.0
            if not pos.is_buy:
                change_pct = -change_pct
            if change_pct <= -stop_loss_pct or change_pct >= take_profit_pct:
                out.append(pos)
        return out
