"""Tests for :mod:`src.strategy.decision_parser`.

Covers the new action shapes the LLM may emit:

- BUY (existing path; size capping).
- CLOSE with optional ``close_fraction`` (the parser resolves it
  against the per-position units table into ``close_units``).
- MODIFY_STOPS (carries SL/TP/trailing, requires bot-owned position).
- HOLD (silently dropped).
- Malformed entries are silently dropped.
"""

from __future__ import annotations

import unittest

from src.config import GuardrailsConfig
from src.etoro.trading import Position
from src.strategy.decision_parser import parse_actions
from src.strategy.signals import Candidate


_GUARDRAILS = GuardrailsConfig(
    max_per_trade_usd=500.0,
    max_parallel_trades=10,
    daily_loss_stop_usd=250.0,
    per_instrument_cooldown_min=60,
    default_stop_loss_pct=5.0,
    default_take_profit_pct=8.0,
    max_leverage=1,
    max_bot_invested_usd=0.0,
    min_amend_remainder_usd=50.0,
)


def _candidate(inst: int, sym: str) -> Candidate:
    return Candidate(
        instrument_id=inst, symbol=sym, action="BUY",
        strength=0.6, reason="test",
        last_close=10.0, rsi=50.0, sma_short=10.0, sma_long=10.0,
        momentum_pct=0.0, raw_score=0.6,
    )


def _position(*, pid: int, iid: int, units: float = 10.0) -> Position:
    return Position(
        position_id=pid, instrument_id=iid, is_buy=True,
        open_rate=10.0, amount=units * 10.0, units=units,
        leverage=1, mirror_id=0, pnl=0.0, raw={},
    )


class BuyParseTests(unittest.TestCase):
    def test_buy_size_capped_to_max_per_trade(self) -> None:
        cands = [_candidate(1, "AAPL")]
        out = parse_actions(
            {"actions": [
                {"action": "BUY", "instrumentId": 1, "amount_usd": 99999},
            ]},
            candidates=cands, bot_owned_positions=[],
            guardrails=_GUARDRAILS,
        )
        assert out is not None
        self.assertEqual(out[0].action, "BUY")
        self.assertEqual(out[0].amount_usd, 500.0)

    def test_buy_zero_falls_back_to_cap(self) -> None:
        cands = [_candidate(1, "AAPL")]
        out = parse_actions(
            {"actions": [
                {"action": "BUY", "instrumentId": 1, "amount_usd": 0},
            ]},
            candidates=cands, bot_owned_positions=[],
            guardrails=_GUARDRAILS,
        )
        assert out is not None
        self.assertEqual(out[0].amount_usd, 500.0)


class CloseParseTests(unittest.TestCase):
    def test_close_resolves_units_from_fraction(self) -> None:
        pos = _position(pid=42, iid=1, units=10.0)
        out = parse_actions(
            {"actions": [
                {"action": "CLOSE", "instrumentId": 1, "positionId": 42,
                 "close_fraction": 0.5},
            ]},
            candidates=[_candidate(1, "AAPL")],
            bot_owned_positions=[pos],
            guardrails=_GUARDRAILS,
            position_units_by_id={42: 10.0},
        )
        assert out is not None
        self.assertEqual(out[0].action, "CLOSE")
        self.assertEqual(out[0].close_fraction, 0.5)
        self.assertEqual(out[0].close_units, 5.0)

    def test_close_full_when_fraction_omitted(self) -> None:
        pos = _position(pid=42, iid=1, units=10.0)
        out = parse_actions(
            {"actions": [
                {"action": "CLOSE", "instrumentId": 1, "positionId": 42},
            ]},
            candidates=[_candidate(1, "AAPL")],
            bot_owned_positions=[pos],
            guardrails=_GUARDRAILS,
            position_units_by_id={42: 10.0},
        )
        assert out is not None
        self.assertIsNone(out[0].close_fraction)
        self.assertIsNone(out[0].close_units)

    def test_close_dropped_when_position_not_owned(self) -> None:
        out = parse_actions(
            {"actions": [
                {"action": "CLOSE", "instrumentId": 1, "positionId": 999},
            ]},
            candidates=[_candidate(1, "AAPL")],
            bot_owned_positions=[],
            guardrails=_GUARDRAILS,
        )
        self.assertEqual(out, [])


class ModifyStopsParseTests(unittest.TestCase):
    def test_modify_stops_parses_all_fields(self) -> None:
        pos = _position(pid=42, iid=1)
        out = parse_actions(
            {"actions": [
                {"action": "MODIFY_STOPS", "instrumentId": 1, "positionId": 42,
                 "stop_loss_pct": 3.0, "take_profit_pct": 7.0,
                 "trailing_stop_pct": 1.5, "rationale": "lock in winner"},
            ]},
            candidates=[_candidate(1, "AAPL")],
            bot_owned_positions=[pos],
            guardrails=_GUARDRAILS,
        )
        assert out is not None
        self.assertEqual(len(out), 1)
        r = out[0]
        self.assertEqual(r.action, "MODIFY_STOPS")
        self.assertEqual(r.stop_loss_pct, 3.0)
        self.assertEqual(r.take_profit_pct, 7.0)
        self.assertEqual(r.trailing_stop_pct, 1.5)
        self.assertEqual(r.rationale, "lock in winner")

    def test_modify_stops_partial_omits_others(self) -> None:
        pos = _position(pid=42, iid=1)
        out = parse_actions(
            {"actions": [
                {"action": "MODIFY_STOPS", "instrumentId": 1, "positionId": 42,
                 "trailing_stop_pct": 2.0},
            ]},
            candidates=[_candidate(1, "AAPL")],
            bot_owned_positions=[pos],
            guardrails=_GUARDRAILS,
        )
        assert out is not None
        r = out[0]
        self.assertIsNone(r.stop_loss_pct)
        self.assertIsNone(r.take_profit_pct)
        self.assertEqual(r.trailing_stop_pct, 2.0)

    def test_modify_stops_with_no_fields_is_dropped(self) -> None:
        pos = _position(pid=42, iid=1)
        out = parse_actions(
            {"actions": [
                {"action": "MODIFY_STOPS", "instrumentId": 1, "positionId": 42},
            ]},
            candidates=[_candidate(1, "AAPL")],
            bot_owned_positions=[pos],
            guardrails=_GUARDRAILS,
        )
        self.assertEqual(out, [])

    def test_modify_stops_for_non_owned_position_dropped(self) -> None:
        out = parse_actions(
            {"actions": [
                {"action": "MODIFY_STOPS", "instrumentId": 1, "positionId": 999,
                 "stop_loss_pct": 3.0},
            ]},
            candidates=[_candidate(1, "AAPL")],
            bot_owned_positions=[],
            guardrails=_GUARDRAILS,
        )
        self.assertEqual(out, [])


class MalformedParseTests(unittest.TestCase):
    def test_holds_dropped(self) -> None:
        out = parse_actions(
            {"actions": [
                {"action": "HOLD", "instrumentId": 1},
            ]},
            candidates=[_candidate(1, "AAPL")],
            bot_owned_positions=[],
            guardrails=_GUARDRAILS,
        )
        self.assertEqual(out, [])

    def test_unparseable_returns_none(self) -> None:
        out = parse_actions(
            "not a dict",  # type: ignore[arg-type]
            candidates=[],
            bot_owned_positions=[],
            guardrails=_GUARDRAILS,
        )
        self.assertIsNone(out)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
