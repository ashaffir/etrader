"""Position monitor — verify, reconcile, and self-heal placed orders.

Responsibilities:

1. **Reconcile**:  after each cycle re-read the portfolio (after the
   eToro 10s cache has settled — caller decides when) and adopt
   newly-opened positions whose ``orderID`` matches one we placed this
   session. CLOSE orders are reconciled by watching the matching
   ``positionID`` disappear from ``snapshot.positions``.

2. **Detect stuck orders**:  delegated to
   :class:`~src.execution.stuck_orders.StuckOrderFinder`. Surfaces a
   list of :class:`StuckOrder` the cycle code can act on.

3. **Auto-cancel + alert**:  delegated to
   :class:`~src.execution.stuck_orders.StuckOrderCanceller`. Cancel
   refusals are re-checked via :func:`get_order_info` to suppress
   race-win alerts (order filled between detection and DELETE).

4. **Synthetic SL/TP** (belt-and-braces): :meth:`positions_needing_close`
   emits close requests when live rate breaches the per-position SL/TP
   bands; only useful if the broker stops enforcing the SL/TP rates we
   pass at placement time.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterable

from ..etoro.client import EtoroClient
from ..etoro.market_data import LiveRate
from ..etoro.trading import Position, PortfolioSnapshot
from ..state import BotState
from ..strategy.tools.base import AssetClass
from .stuck_orders import (
    CancelResult,
    StuckOrder,
    StuckOrderCanceller,
    StuckOrderFinder,
)
from .tracked_order import TrackedOrder


class PositionMonitor:
    """Stateful per-session order tracker (one instance per bot process)."""

    def __init__(
        self,
        *,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
    ) -> None:
        self._tracked: dict[int, TrackedOrder] = {}
        self._log = logger or logging.getLogger("etrader.execution.monitor")
        self._finder = StuckOrderFinder()
        self._canceller = StuckOrderCanceller(
            on_settled=self._evict,
            logger=self._log,
        )

    # ------------------------------------------------------------------
    # Recording placements
    # ------------------------------------------------------------------

    def track_open(
        self,
        *,
        order_id: int,
        instrument_id: int,
        symbol: str,
        asset_class: AssetClass,
        amount_usd: float,
        placed_at_utc: datetime,
        placed_at_monotonic: float,
    ) -> None:
        self._tracked[order_id] = TrackedOrder(
            order_id=order_id,
            instrument_id=instrument_id,
            symbol=symbol,
            action="BUY",
            asset_class=asset_class,
            amount_usd=amount_usd,
            placed_at_utc=placed_at_utc,
            placed_at_monotonic=placed_at_monotonic,
        )

    def track_close(
        self,
        *,
        order_id: int,
        position_id: int,
        instrument_id: int,
        symbol: str,
        asset_class: AssetClass,
        placed_at_utc: datetime,
        placed_at_monotonic: float,
    ) -> None:
        self._tracked[order_id] = TrackedOrder(
            order_id=order_id,
            instrument_id=instrument_id,
            symbol=symbol,
            action="CLOSE",
            asset_class=asset_class,
            amount_usd=0.0,
            placed_at_utc=placed_at_utc,
            placed_at_monotonic=placed_at_monotonic,
            position_id=position_id,
        )

    @property
    def tracked_count(self) -> int:
        return len(self._tracked)

    def tracked_for(self, order_id: int) -> TrackedOrder | None:
        return self._tracked.get(order_id)

    def tracked_orders(self) -> list[TrackedOrder]:
        return list(self._tracked.values())

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    def reconcile(self, snapshot: PortfolioSnapshot, state: BotState) -> list[Position]:
        """Match tracked orders against the live portfolio.

        BUY tracked orders settle when a position with the same
        ``orderID`` appears in ``snapshot.positions``. CLOSE tracked
        orders settle when their matching ``position_id`` is no longer
        present.
        """
        if not self._tracked:
            return []

        adopted: list[Position] = []
        snapshot_position_ids: set[int] = {p.position_id for p in snapshot.positions}
        position_by_order: dict[int, Position] = {}
        for pos in snapshot.positions:
            ord_id = int(pos.raw.get("orderID") or pos.raw.get("orderId") or 0)
            if ord_id:
                position_by_order[ord_id] = pos

        for order_id in list(self._tracked.keys()):
            tracked = self._tracked[order_id]
            if tracked.action == "BUY":
                pos = position_by_order.get(order_id)
                if pos is not None:
                    state.add_owned(pos.position_id)
                    adopted.append(pos)
                    self._evict(order_id)
            elif tracked.action == "CLOSE":
                if tracked.position_id not in snapshot_position_ids:
                    self._evict(order_id)

        if adopted:
            self._log.info(
                "[monitor] adopted %d new bot-owned position(s)", len(adopted),
            )
        return adopted

    # ------------------------------------------------------------------
    # Stuck-order pipeline
    # ------------------------------------------------------------------

    def find_stuck(
        self,
        snapshot: PortfolioSnapshot,
        *,
        now_utc: datetime,
        grace_seconds_after_open: int,
    ) -> list[StuckOrder]:
        """Return tracked orders that should have filled by now."""
        if not self._tracked:
            return []
        return self._finder.find(
            tracked=self._tracked.values(),
            snapshot=snapshot,
            now_utc=now_utc,
            grace_seconds_after_open=grace_seconds_after_open,
        )

    def cancel(
        self,
        client: EtoroClient,
        *,
        env: str,
        stuck: StuckOrder,
    ) -> CancelResult:
        """Cancel one stuck order; suppresses alerts on race-wins."""
        return self._canceller.cancel(client, env=env, stuck=stuck)

    # ------------------------------------------------------------------
    # Synthetic SL/TP
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

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _evict(self, order_id: int) -> None:
        self._tracked.pop(order_id, None)
