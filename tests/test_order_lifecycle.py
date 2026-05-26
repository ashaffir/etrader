"""Tests for src.etoro.order_lifecycle — status parsing + cancel wrappers."""

from __future__ import annotations

import unittest
from typing import Any

from src.etoro.order_lifecycle import (
    OrderInfo,
    OrderStatus,
    cancel_market_close_order,
    cancel_market_open_order,
    get_order_info,
)


class OrderStatusTests(unittest.TestCase):
    def test_known_values_map(self) -> None:
        for raw, expected in (
            (0, OrderStatus.PENDING),
            (1, OrderStatus.EXECUTED),
            (2, OrderStatus.CANCELLED),
            (3, OrderStatus.REJECTED),
            (4, OrderStatus.PARTIALLY_EXECUTED),
        ):
            self.assertEqual(OrderStatus.from_raw(raw), expected)

    def test_unknown_falls_back_safely(self) -> None:
        self.assertEqual(OrderStatus.from_raw(None), OrderStatus.UNKNOWN)
        self.assertEqual(OrderStatus.from_raw("garbage"), OrderStatus.UNKNOWN)
        self.assertEqual(OrderStatus.from_raw(99), OrderStatus.UNKNOWN)

    def test_terminal_classification(self) -> None:
        self.assertTrue(OrderStatus.EXECUTED.is_terminal)
        self.assertTrue(OrderStatus.CANCELLED.is_terminal)
        self.assertTrue(OrderStatus.REJECTED.is_terminal)
        self.assertFalse(OrderStatus.PENDING.is_terminal)
        self.assertFalse(OrderStatus.PARTIALLY_EXECUTED.is_terminal)
        self.assertFalse(OrderStatus.UNKNOWN.is_terminal)

    def test_pending_classification(self) -> None:
        self.assertTrue(OrderStatus.PENDING.is_pending)
        self.assertTrue(OrderStatus.UNKNOWN.is_pending)  # safe default
        self.assertFalse(OrderStatus.EXECUTED.is_pending)


class _FakeClient:
    def __init__(self) -> None:
        self.delete_calls: list[str] = []
        self.get_calls: list[str] = []
        self.next_get: dict[str, Any] = {}

    def delete(self, path: str) -> dict[str, Any]:
        self.delete_calls.append(path)
        return {"token": "xyz"}

    def get(self, path: str, params: Any = None, *, retries: int = 0) -> Any:
        self.get_calls.append(path)
        return self.next_get


class CancelWrappersTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _FakeClient()

    def test_cancel_market_open_demo(self) -> None:
        result = cancel_market_open_order(self.client, env="demo", order_id=7001)
        self.assertEqual(result, {"token": "xyz"})
        self.assertEqual(
            self.client.delete_calls,
            ["/trading/execution/demo/market-open-orders/7001"],
        )

    def test_cancel_market_open_real(self) -> None:
        cancel_market_open_order(self.client, env="real", order_id=7002)
        self.assertEqual(
            self.client.delete_calls,
            ["/trading/execution/real/market-open-orders/7002"],
        )

    def test_cancel_market_close_demo(self) -> None:
        cancel_market_close_order(self.client, env="demo", order_id=8003)
        self.assertEqual(
            self.client.delete_calls,
            ["/trading/execution/demo/market-close-orders/8003"],
        )

    def test_cancel_rejects_invalid_env(self) -> None:
        with self.assertRaises(ValueError):
            cancel_market_open_order(self.client, env="prod", order_id=1)
        with self.assertRaises(ValueError):
            cancel_market_close_order(self.client, env="staging", order_id=1)


class GetOrderInfoTests(unittest.TestCase):
    def test_parses_capital_id_payload(self) -> None:
        client = _FakeClient()
        client.next_get = {
            "orderID": 123,
            "statusID": 1,
            "errorCode": None,
            "errorMessage": None,
            "instrumentID": 67890,
            "amount": 1000.0,
        }
        info = get_order_info(client, "demo", 123)
        self.assertEqual(info.order_id, 123)
        self.assertEqual(info.status, OrderStatus.EXECUTED)
        self.assertEqual(info.instrument_id, 67890)
        self.assertEqual(info.amount_usd, 1000.0)
        self.assertFalse(info.has_error)

    def test_parses_lowercamel_fallback(self) -> None:
        """The API spec uses capital IDs but we tolerate either."""
        client = _FakeClient()
        client.next_get = {
            "orderId": 456,
            "statusId": 0,
        }
        info = get_order_info(client, "demo", 456)
        self.assertEqual(info.order_id, 456)
        self.assertEqual(info.status, OrderStatus.PENDING)

    def test_has_error_flips_on_error_fields(self) -> None:
        client = _FakeClient()
        client.next_get = {
            "orderID": 9,
            "statusID": 3,
            "errorCode": 42,
            "errorMessage": "insufficient cash",
        }
        info = get_order_info(client, "demo", 9)
        self.assertTrue(info.has_error)
        self.assertEqual(info.status, OrderStatus.REJECTED)

    def test_empty_response_yields_unknown_status(self) -> None:
        client = _FakeClient()
        client.next_get = {}
        info = get_order_info(client, "demo", 1)
        self.assertEqual(info.status, OrderStatus.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
