"""Stuck-order detection + cancellation pipeline.

Split out of :mod:`monitor` to keep that module under the per-file
size guideline. The flow is:

1. :class:`StuckOrderFinder.find` filters the monitor's tracked orders
   down to ones that should have filled by now (session-aware grace).
2. :class:`StuckOrderCanceller.cancel` issues the DELETE against eToro
   and decides whether the result warrants a Telegram alert (failure
   case where the order is still genuinely unsettled).

Both classes are thin coordinators over the :mod:`session` helpers
and the :mod:`src.etoro.order_lifecycle` API wrappers, so they're
easy to unit-test without standing up a full ``PositionMonitor``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable

from ..etoro.client import EtoroClient
from ..etoro.errors import EtoroApiError
from ..etoro.order_lifecycle import (
    OrderStatus,
    cancel_market_close_order,
    cancel_market_open_order,
    get_order_info,
)
from ..etoro.trading import PortfolioSnapshot
from ..strategy.tools.base import AssetClass
from .session import session_state
from .tracked_order import TrackedOrder


@dataclass(frozen=True)
class StuckOrder:
    """A tracked order that has overshot its grace window."""

    tracked: TrackedOrder
    waited_seconds: int
    in_session_seconds: int


@dataclass(frozen=True)
class CancelResult:
    """Outcome of one cancel attempt.

    ``alert=True`` means the cycle should emit
    :data:`AlertType.ORDER_STUCK_CANT_CANCEL`. A "cancel failed because
    the order filled" race is suppressed — the right outcome happened.
    """

    tracked: TrackedOrder
    cancelled: bool
    alert: bool
    detail: str
    final_status: OrderStatus | None = None


# ---------------------------------------------------------------------------
# Stuck detection
# ---------------------------------------------------------------------------

class StuckOrderFinder:
    """Pure logic: decide which tracked orders are "stuck"."""

    def find(
        self,
        *,
        tracked: Iterable[TrackedOrder],
        snapshot: PortfolioSnapshot,
        now_utc: datetime,
        grace_seconds_after_open: int,
    ) -> list[StuckOrder]:
        pending_open_ids: set[int] = {o.order_id for o in snapshot.orders_for_open}
        stuck: list[StuckOrder] = []
        for t in tracked:
            if not self._eligible(t, now_utc=now_utc, grace_seconds=grace_seconds_after_open):
                continue
            if t.action == "BUY" and t.order_id not in pending_open_ids:
                # Not yet visible on the broker side AND not adopted —
                # transient gap between place and next /pnl fetch.
                continue
            waited = int((now_utc - t.placed_at_utc).total_seconds())
            stuck.append(StuckOrder(
                tracked=t,
                waited_seconds=waited,
                in_session_seconds=_in_session_seconds(t.asset_class, now_utc, waited),
            ))
        return stuck

    @staticmethod
    def _eligible(
        tracked: TrackedOrder, *, now_utc: datetime, grace_seconds: int,
    ) -> bool:
        waited = int((now_utc - tracked.placed_at_utc).total_seconds())
        if waited < grace_seconds:
            return False
        state = session_state(tracked.asset_class, now_utc)
        if not state.is_open:
            return False
        if state.seconds_since_open is None:
            # Always-open market (crypto) — absolute grace already met.
            return True
        return state.seconds_since_open >= grace_seconds


def _in_session_seconds(
    asset_class: AssetClass, now_utc: datetime, waited_seconds: int,
) -> int:
    state = session_state(asset_class, now_utc)
    if state.seconds_since_open is None:
        return waited_seconds
    return min(waited_seconds, state.seconds_since_open)


# ---------------------------------------------------------------------------
# Cancellation + race-condition handling
# ---------------------------------------------------------------------------

class StuckOrderCanceller:
    """Issues cancel calls + decides whether the outcome warrants an alert.

    The ``on_settled`` callback lets the owning monitor evict the
    order from its tracked map when the cancel either succeeded or
    confirmed a terminal state via the recheck.
    """

    def __init__(
        self,
        *,
        on_settled: Callable[[int], None],
        logger: logging.Logger | logging.LoggerAdapter | None = None,
    ) -> None:
        self._on_settled = on_settled
        self._log = logger or logging.getLogger("etrader.execution.stuck")

    def cancel(
        self,
        client: EtoroClient,
        *,
        env: str,
        stuck: StuckOrder,
    ) -> CancelResult:
        tracked = stuck.tracked
        try:
            if tracked.action == "BUY":
                cancel_market_open_order(client, env=env, order_id=tracked.order_id)
            else:
                cancel_market_close_order(client, env=env, order_id=tracked.order_id)
        except EtoroApiError as exc:
            return self._handle_failure(client, env=env, tracked=tracked, exc=exc)

        self._on_settled(tracked.order_id)
        self._log.warning(
            "[monitor] cancelled stuck %s %s (orderID=%d, waited %ds, in-session %ds)",
            tracked.action, tracked.symbol, tracked.order_id,
            stuck.waited_seconds, stuck.in_session_seconds,
        )
        return CancelResult(
            tracked=tracked,
            cancelled=True,
            alert=False,
            detail=f"cancelled after {stuck.waited_seconds}s",
        )

    def _handle_failure(
        self,
        client: EtoroClient,
        *,
        env: str,
        tracked: TrackedOrder,
        exc: EtoroApiError,
    ) -> CancelResult:
        try:
            info = get_order_info(client, env, tracked.order_id)
            final_status: OrderStatus | None = info.status
        except EtoroApiError as recheck_exc:
            self._log.warning(
                "[monitor] cancel of %s #%d failed AND status recheck failed: %s / %s",
                tracked.symbol, tracked.order_id, exc, recheck_exc,
            )
            return CancelResult(
                tracked=tracked,
                cancelled=False,
                alert=True,
                detail=f"cancel refused ({exc}); status recheck also failed ({recheck_exc})",
                final_status=None,
            )

        if final_status == OrderStatus.EXECUTED:
            self._on_settled(tracked.order_id)
            self._log.info(
                "[monitor] cancel of %s #%d refused — order already EXECUTED. Standing down.",
                tracked.symbol, tracked.order_id,
            )
            return CancelResult(
                tracked=tracked,
                cancelled=False,
                alert=False,
                detail="order executed before cancel landed",
                final_status=final_status,
            )

        if final_status == OrderStatus.CANCELLED:
            self._on_settled(tracked.order_id)
            self._log.info(
                "[monitor] cancel of %s #%d refused — order already CANCELLED.",
                tracked.symbol, tracked.order_id,
            )
            return CancelResult(
                tracked=tracked,
                cancelled=False,
                alert=False,
                detail="order already cancelled",
                final_status=final_status,
            )

        # PENDING / REJECTED / PARTIALLY_EXECUTED / UNKNOWN — operator action.
        self._log.error(
            "[monitor] cancel of %s #%d REFUSED and status=%s (%s)",
            tracked.symbol, tracked.order_id, final_status.name, exc,
        )
        return CancelResult(
            tracked=tracked,
            cancelled=False,
            alert=True,
            detail=f"cancel refused ({exc}); status={final_status.name}",
            final_status=final_status,
        )

