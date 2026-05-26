"""Tests for the stuck-order detection + cancel pipeline."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from typing import Any

from src.etoro.errors import EtoroApiError, EtoroBadRequestError
from src.etoro.order_lifecycle import OrderInfo, OrderStatus
from src.etoro.trading import PendingOpenOrder, PortfolioSnapshot
from src.execution.monitor import PositionMonitor
from src.execution.stuck_orders import StuckOrderCanceller, StuckOrderFinder
from src.execution.tracked_order import TrackedOrder
from src.strategy.tools.base import AssetClass


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


def _pending_open(order_id: int, instrument_id: int = 10) -> PendingOpenOrder:
    return PendingOpenOrder(
        order_id=order_id,
        instrument_id=instrument_id,
        amount=100.0,
        is_buy=True,
        leverage=1,
        mirror_id=0,
        raw={},
    )


def _empty_snapshot(orders_for_open: list[PendingOpenOrder] | None = None) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        credit=10_000.0,
        unrealized_pnl=0.0,
        positions=[],
        orders=[],
        orders_for_open=list(orders_for_open or []),
        mirrors=[],
        raw={},
    )


def _tracked(
    *,
    order_id: int = 7001,
    asset_class: AssetClass = AssetClass.STOCK,
    action: str = "BUY",
    placed_at_utc: datetime,
    position_id: int = 0,
) -> TrackedOrder:
    return TrackedOrder(
        order_id=order_id,
        instrument_id=1,
        symbol="ACME",
        action=action,
        asset_class=asset_class,
        amount_usd=100.0,
        placed_at_utc=placed_at_utc,
        placed_at_monotonic=0.0,
        position_id=position_id,
    )


# ---------------------------------------------------------------------------
# StuckOrderFinder
# ---------------------------------------------------------------------------

class FinderTests(unittest.TestCase):
    """Pure logic: which tracked orders should we declare stuck?"""

    def test_stock_inside_session_past_grace(self) -> None:
        placed = _utc(2026, 5, 26, 14, 0)
        now = _utc(2026, 5, 26, 14, 10)  # 10 minutes after placement
        snap = _empty_snapshot([_pending_open(7001)])
        out = StuckOrderFinder().find(
            tracked=[_tracked(order_id=7001, placed_at_utc=placed)],
            snapshot=snap,
            now_utc=now,
            grace_seconds_after_open=300,  # 5 minutes
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].waited_seconds, 600)

    def test_stock_outside_session_never_stuck(self) -> None:
        """Order placed pre-market should not be considered stuck."""
        placed = _utc(2026, 5, 26, 7, 0)
        now = _utc(2026, 5, 26, 8, 0)  # still pre-market (open is 13:30)
        snap = _empty_snapshot([_pending_open(7001)])
        out = StuckOrderFinder().find(
            tracked=[_tracked(order_id=7001, placed_at_utc=placed)],
            snapshot=snap,
            now_utc=now,
            grace_seconds_after_open=60,
        )
        self.assertEqual(out, [])

    def test_stock_just_opened_within_grace_not_stuck(self) -> None:
        """If market just opened and grace hasn't elapsed in-session yet, skip."""
        placed = _utc(2026, 5, 26, 7, 0)
        now = _utc(2026, 5, 26, 13, 32)  # 2 minutes after equity open
        snap = _empty_snapshot([_pending_open(7001)])
        out = StuckOrderFinder().find(
            tracked=[_tracked(order_id=7001, placed_at_utc=placed)],
            snapshot=snap,
            now_utc=now,
            grace_seconds_after_open=300,
        )
        self.assertEqual(out, [])

    def test_stock_in_session_after_grace_is_stuck(self) -> None:
        placed = _utc(2026, 5, 26, 7, 0)
        now = _utc(2026, 5, 26, 13, 40)  # 10 min after equity open
        snap = _empty_snapshot([_pending_open(7001)])
        out = StuckOrderFinder().find(
            tracked=[_tracked(order_id=7001, placed_at_utc=placed)],
            snapshot=snap,
            now_utc=now,
            grace_seconds_after_open=300,
        )
        self.assertEqual(len(out), 1)

    def test_crypto_uses_absolute_grace(self) -> None:
        """Crypto is always-open: stuck = absolute age > grace, regardless of clock."""
        placed = _utc(2026, 5, 24, 3, 0)
        now = placed + timedelta(minutes=10)
        snap = _empty_snapshot([_pending_open(7001)])
        out = StuckOrderFinder().find(
            tracked=[_tracked(
                order_id=7001,
                asset_class=AssetClass.CRYPTO,
                placed_at_utc=placed,
            )],
            snapshot=snap,
            now_utc=now,
            grace_seconds_after_open=300,
        )
        self.assertEqual(len(out), 1)

    def test_open_order_not_in_snapshot_orders_skipped(self) -> None:
        """BUY orders must appear in snapshot.orders_for_open to be 'stuck'."""
        placed = _utc(2026, 5, 26, 13, 0)
        now = _utc(2026, 5, 26, 14, 0)
        # Empty orders_for_open — broker doesn't show it as pending yet.
        snap = _empty_snapshot([])
        out = StuckOrderFinder().find(
            tracked=[_tracked(order_id=7001, placed_at_utc=placed)],
            snapshot=snap,
            now_utc=now,
            grace_seconds_after_open=300,
        )
        self.assertEqual(out, [])

    def test_close_order_doesnt_require_orders_for_open(self) -> None:
        """CLOSE-side stuck detection works without orders_for_open visibility."""
        placed = _utc(2026, 5, 26, 13, 35)
        now = _utc(2026, 5, 26, 13, 45)
        snap = _empty_snapshot([])  # no relevant entries
        out = StuckOrderFinder().find(
            tracked=[_tracked(
                order_id=9001,
                action="CLOSE",
                placed_at_utc=placed,
                position_id=12345,
            )],
            snapshot=snap,
            now_utc=now,
            grace_seconds_after_open=300,
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].tracked.action, "CLOSE")


# ---------------------------------------------------------------------------
# StuckOrderCanceller — fakes / spies for the HTTP layer
# ---------------------------------------------------------------------------

class _FakeClient:
    """In-memory stand-in for EtoroClient that records calls.

    ``next_delete_raises`` lets a test queue exception responses; the
    queue is consumed in FIFO order. ``next_get_returns`` does the
    same for ``get_order_info`` results.
    """

    def __init__(self) -> None:
        self.delete_calls: list[str] = []
        self.get_calls: list[str] = []
        self.next_delete_raises: list[Exception | None] = []
        self.next_get_returns: list[dict[str, Any] | Exception] = []

    def delete(self, path: str) -> dict[str, Any]:
        self.delete_calls.append(path)
        if self.next_delete_raises:
            queued = self.next_delete_raises.pop(0)
            if queued is not None:
                raise queued
        return {"token": "ok"}

    def get(self, path: str, params: Any = None, *, retries: int = 0) -> Any:
        self.get_calls.append(path)
        if not self.next_get_returns:
            return {}
        queued = self.next_get_returns.pop(0)
        if isinstance(queued, Exception):
            raise queued
        return queued


def _stuck(order_id: int = 7001, action: str = "BUY") -> Any:
    """Build a StuckOrder for canceller-level tests."""
    from src.execution.stuck_orders import StuckOrder

    return StuckOrder(
        tracked=_tracked(
            order_id=order_id,
            action=action,
            placed_at_utc=_utc(2026, 5, 26, 13, 0),
            position_id=12345 if action == "CLOSE" else 0,
        ),
        waited_seconds=600,
        in_session_seconds=300,
    )


class CancellerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _FakeClient()
        self.evicted: list[int] = []
        self.canceller = StuckOrderCanceller(on_settled=self.evicted.append)

    def test_open_cancel_hits_open_endpoint(self) -> None:
        result = self.canceller.cancel(self.client, env="demo", stuck=_stuck())
        self.assertTrue(result.cancelled)
        self.assertFalse(result.alert)
        self.assertEqual(
            self.client.delete_calls,
            ["/trading/execution/demo/market-open-orders/7001"],
        )
        self.assertIn(7001, self.evicted)

    def test_close_cancel_hits_close_endpoint(self) -> None:
        result = self.canceller.cancel(
            self.client, env="real", stuck=_stuck(order_id=8002, action="CLOSE"),
        )
        self.assertTrue(result.cancelled)
        self.assertEqual(
            self.client.delete_calls,
            ["/trading/execution/real/market-close-orders/8002"],
        )

    def test_cancel_refused_but_order_executed_no_alert(self) -> None:
        """Race-and-lose: cancel 4xx + order shows EXECUTED → no alert."""
        self.client.next_delete_raises = [EtoroBadRequestError("already done", 400)]
        self.client.next_get_returns = [{
            "orderID": 7001,
            "statusID": int(OrderStatus.EXECUTED),
            "instrumentID": 1,
            "amount": 100.0,
        }]
        result = self.canceller.cancel(self.client, env="demo", stuck=_stuck())
        self.assertFalse(result.cancelled)
        self.assertFalse(result.alert)
        self.assertEqual(result.final_status, OrderStatus.EXECUTED)
        self.assertIn(7001, self.evicted)

    def test_cancel_refused_order_cancelled_externally_no_alert(self) -> None:
        """Operator cancelled in /app: status=2 → no alert."""
        self.client.next_delete_raises = [EtoroBadRequestError("nope", 400)]
        self.client.next_get_returns = [{
            "orderID": 7001,
            "statusID": int(OrderStatus.CANCELLED),
        }]
        result = self.canceller.cancel(self.client, env="demo", stuck=_stuck())
        self.assertFalse(result.cancelled)
        self.assertFalse(result.alert)
        self.assertIn(7001, self.evicted)

    def test_cancel_refused_order_still_pending_alerts(self) -> None:
        """Cancel refused AND order still PENDING → genuine stuck → alert."""
        self.client.next_delete_raises = [EtoroBadRequestError("nope", 400)]
        self.client.next_get_returns = [{
            "orderID": 7001,
            "statusID": int(OrderStatus.PENDING),
        }]
        result = self.canceller.cancel(self.client, env="demo", stuck=_stuck())
        self.assertFalse(result.cancelled)
        self.assertTrue(result.alert)
        self.assertEqual(result.final_status, OrderStatus.PENDING)
        # Should NOT evict — operator may want to retry by hand.
        self.assertNotIn(7001, self.evicted)

    def test_cancel_refused_rejected_order_alerts(self) -> None:
        """Rejected orders need operator attention (likely a config issue)."""
        self.client.next_delete_raises = [EtoroBadRequestError("nope", 400)]
        self.client.next_get_returns = [{
            "orderID": 7001,
            "statusID": int(OrderStatus.REJECTED),
            "errorCode": 42,
            "errorMessage": "insufficient cash",
        }]
        result = self.canceller.cancel(self.client, env="demo", stuck=_stuck())
        self.assertFalse(result.cancelled)
        self.assertTrue(result.alert)
        self.assertEqual(result.final_status, OrderStatus.REJECTED)

    def test_cancel_refused_and_recheck_fails_alerts(self) -> None:
        """If we can't even fetch status after cancel-refusal, alert."""
        self.client.next_delete_raises = [EtoroBadRequestError("nope", 400)]
        self.client.next_get_returns = [EtoroApiError("network down")]
        result = self.canceller.cancel(self.client, env="demo", stuck=_stuck())
        self.assertFalse(result.cancelled)
        self.assertTrue(result.alert)
        self.assertIsNone(result.final_status)


# ---------------------------------------------------------------------------
# PositionMonitor integration: reconcile + find_stuck + cancel
# ---------------------------------------------------------------------------

class MonitorReconcileCloseTests(unittest.TestCase):
    """CLOSE orders settle when the matching position vanishes from the snapshot."""

    def test_close_reconciles_when_position_disappears(self) -> None:
        monitor = PositionMonitor()
        monitor.track_close(
            order_id=9001,
            position_id=12345,
            instrument_id=1,
            symbol="ACME",
            asset_class=AssetClass.STOCK,
            placed_at_utc=_utc(2026, 5, 26, 14, 0),
            placed_at_monotonic=0.0,
        )
        self.assertEqual(monitor.tracked_count, 1)
        snap = _empty_snapshot()  # no positions → close has settled
        monitor.reconcile(snap, state=_FakeState())
        self.assertEqual(monitor.tracked_count, 0)


class _FakeState:
    """Minimal BotState stand-in: only ``add_owned`` is called from reconcile."""

    def __init__(self) -> None:
        self.owned: set[int] = set()

    def add_owned(self, pid: int) -> None:
        self.owned.add(pid)


if __name__ == "__main__":
    unittest.main()
