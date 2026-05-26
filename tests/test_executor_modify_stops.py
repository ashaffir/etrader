"""Integration tests for MODIFY_STOPS + partial CLOSE in the executor.

We use a minimal fake eToro client + bot state to assert that:

- MODIFY_STOPS writes to the DynamicStopsStore and never hits the broker.
- Partial close passes ``units_to_deduct`` to ``close_position_by_market``.
- Full close clears the dynamic-stops entry for the closed position.
"""

from __future__ import annotations

import logging
import unittest
from typing import Any

from src.config import GuardrailsConfig, OperationsConfig
from src.execution.dynamic_stops import DynamicStopsStore
from src.execution.executor import TradeExecutor
from src.state import BotState
from src.strategy.risk import TradeRequest, TradeVerdict


class _FakeClient:
    """Captures the last call so the tests can assert on it."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def post(self, path: str, *, json: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("POST", path, json))
        return {
            "orderForClose": {"orderID": 9001},
            "orderForOpen": {"orderID": 9002},
        }


def _guardrails() -> GuardrailsConfig:
    return GuardrailsConfig(
        max_per_trade_usd=500.0, max_parallel_trades=10,
        daily_loss_stop_usd=250.0, per_instrument_cooldown_min=60,
        default_stop_loss_pct=5.0, default_take_profit_pct=8.0,
        max_leverage=1, max_bot_invested_usd=0.0,
        min_amend_remainder_usd=50.0,
    )


def _build_executor() -> tuple[TradeExecutor, DynamicStopsStore, _FakeClient]:
    client = _FakeClient()
    store = DynamicStopsStore(
        default_stop_loss_pct=5.0, default_take_profit_pct=8.0,
    )
    ex = TradeExecutor(
        client=client,  # type: ignore[arg-type]
        env="demo",
        guardrails=_guardrails(),
        operations=OperationsConfig(
            check_interval_seconds=60,
            universe_refresh_minutes=30,
            candle_interval="OneHour", candle_count=100,
            request_timeout_seconds=20, trade_spacing_seconds=0,
            pending_grace_seconds_after_open=300,
            cancel_stuck_orders_enabled=True,
        ),
        logger=logging.getLogger("test"),
        dynamic_stops=store,
    )
    return ex, store, client


class ModifyStopsExecutionTests(unittest.TestCase):
    def test_modify_stops_writes_to_store_and_does_not_call_broker(self) -> None:
        ex, store, client = _build_executor()
        state = BotState()
        state.add_owned(42)
        verdict = TradeVerdict(
            request=TradeRequest(
                instrument_id=1, symbol="AAPL", action="MODIFY_STOPS",
                amount_usd=0.0, position_id=42,
                stop_loss_pct=3.0, take_profit_pct=7.0,
                trailing_stop_pct=1.5, rationale="lock in",
            ),
            approved=True, reason="approved (MODIFY_STOPS)",
        )
        results = ex.execute_all(verdicts=[verdict], rates={}, state=state)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, "ok")
        self.assertEqual(results[0].action, "MODIFY_STOPS")
        # No broker call:
        self.assertEqual(client.calls, [])
        # Store updated:
        band = store.effective_band(42)
        self.assertEqual(band.stop_loss_pct, 3.0)
        self.assertEqual(band.take_profit_pct, 7.0)
        self.assertEqual(band.trailing_stop_pct, 1.5)
        self.assertEqual(band.rationale, "lock in")

    def test_modify_stops_skipped_when_store_missing(self) -> None:
        # Build an executor WITHOUT a dynamic-stops store.
        client = _FakeClient()
        ex = TradeExecutor(
            client=client,  # type: ignore[arg-type]
            env="demo",
            guardrails=_guardrails(),
            operations=OperationsConfig(
                check_interval_seconds=60, universe_refresh_minutes=30,
                candle_interval="OneHour", candle_count=100,
                request_timeout_seconds=20, trade_spacing_seconds=0,
                pending_grace_seconds_after_open=300,
                cancel_stuck_orders_enabled=True,
            ),
            logger=logging.getLogger("test"),
            # no dynamic_stops kwarg
        )
        state = BotState()
        state.add_owned(42)
        verdict = TradeVerdict(
            request=TradeRequest(
                instrument_id=1, symbol="AAPL", action="MODIFY_STOPS",
                amount_usd=0.0, position_id=42, stop_loss_pct=3.0,
            ),
            approved=True, reason="approved",
        )
        results = ex.execute_all(verdicts=[verdict], rates={}, state=state)
        self.assertEqual(results[0].status, "skipped")
        self.assertEqual(client.calls, [])


class PartialCloseExecutionTests(unittest.TestCase):
    def test_partial_close_passes_units_to_deduct(self) -> None:
        ex, store, client = _build_executor()
        state = BotState()
        state.add_owned(42)
        verdict = TradeVerdict(
            request=TradeRequest(
                instrument_id=1, symbol="AAPL", action="CLOSE",
                amount_usd=0.0, position_id=42,
                close_fraction=0.5, close_units=5.0,
            ),
            approved=True, reason="approved",
        )
        results = ex.execute_all(verdicts=[verdict], rates={}, state=state)
        self.assertEqual(results[0].status, "ok")
        self.assertEqual(len(client.calls), 1)
        _, path, body = client.calls[0]
        self.assertIn("market-close-orders", path)
        self.assertEqual(body.get("UnitsToDeduct"), 5.0)
        # Partial close: position remains owned, stops kept.
        self.assertIn(42, state.bot_owned_positions)

    def test_full_close_omits_units_and_clears_state(self) -> None:
        ex, store, client = _build_executor()
        state = BotState()
        state.add_owned(42)
        store.set_band(42, stop_loss_pct=3.0)
        verdict = TradeVerdict(
            request=TradeRequest(
                instrument_id=1, symbol="AAPL", action="CLOSE",
                amount_usd=0.0, position_id=42,
            ),
            approved=True, reason="approved",
        )
        ex.execute_all(verdicts=[verdict], rates={}, state=state)
        _, path, body = client.calls[0]
        self.assertNotIn("UnitsToDeduct", body)
        # Full close: position de-registered + dynamic stops cleared.
        self.assertNotIn(42, state.bot_owned_positions)
        self.assertFalse(store.has_override(42))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
