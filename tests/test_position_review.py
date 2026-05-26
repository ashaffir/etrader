"""Regression tests for :mod:`src.strategy.position_review`."""

from __future__ import annotations

import time
import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from src.etoro.trading import Position
from src.strategy.position_review import (
    PositionReviewer,
    PositionReviewConfig,
)


@dataclass
class _OpenSnap:
    """Minimal mock of the tracker's OpenTradeState — only the fields the
    reviewer reads."""
    last_pnl_usd: float
    last_pnl_pct: float
    mfe_usd: float
    mae_usd: float
    opened_at_iso: str

    @classmethod
    def fresh(cls, *, pnl_usd: float, pnl_pct: float, mfe: float = 0.0,
              mae: float = 0.0, opened_ago_seconds: float = 30.0) -> "_OpenSnap":
        opened = datetime.now(timezone.utc) - timedelta(seconds=opened_ago_seconds)
        return cls(
            last_pnl_usd=pnl_usd, last_pnl_pct=pnl_pct,
            mfe_usd=mfe, mae_usd=mae,
            opened_at_iso=opened.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )


def _pos(*, pid: int = 100, iid: int = 1, amount: float = 100.0,
         open_rate: float = 10.0, is_buy: bool = True, pnl: float = 0.0,
         mirror: int = 0) -> Position:
    return Position(
        position_id=pid, instrument_id=iid, is_buy=is_buy,
        open_rate=open_rate, amount=amount, units=amount / open_rate,
        leverage=1, mirror_id=mirror, pnl=pnl, raw={},
    )


def _reviewer(**overrides) -> PositionReviewer:
    cfg = PositionReviewConfig(
        drawdown_pct=overrides.get("drawdown_pct", 2.0),
        pullback_pct=overrides.get("pullback_pct", 3.0),
        stale_hold_minutes=overrides.get("stale_hold_minutes", 60.0),
        stale_threshold_pct=overrides.get("stale_threshold_pct", 0.5),
        max_hold_minutes=overrides.get("max_hold_minutes", 240.0),
    )
    return PositionReviewer(cfg)


class DrawdownTriggerTests(unittest.TestCase):
    def test_below_threshold_fires(self) -> None:
        pos = _pos(pid=1)
        snap = _OpenSnap.fresh(pnl_usd=-5.0, pnl_pct=-2.5)
        out = _reviewer().evaluate(
            bot_owned_positions=[pos], symbol_for_id={1: "AAPL"},
            perf_open_states={1: snap},
        )
        self.assertEqual(len(out), 1)
        self.assertIn("drawdown", out[0].triggers)

    def test_above_threshold_does_not_fire(self) -> None:
        pos = _pos(pid=1)
        snap = _OpenSnap.fresh(pnl_usd=-1.0, pnl_pct=-0.5)
        out = _reviewer(drawdown_pct=2.0).evaluate(
            bot_owned_positions=[pos], symbol_for_id={1: "AAPL"},
            perf_open_states={1: snap},
        )
        self.assertEqual(out, [])


class PullbackTriggerTests(unittest.TestCase):
    def test_gave_back_half_of_mfe(self) -> None:
        # MFE=10, current=4 → pullback of 6 = 60% of MFE.
        pos = _pos(pid=1)
        snap = _OpenSnap.fresh(pnl_usd=4.0, pnl_pct=4.0, mfe=10.0)
        out = _reviewer(pullback_pct=50.0).evaluate(
            bot_owned_positions=[pos], symbol_for_id={1: "AAPL"},
            perf_open_states={1: snap},
        )
        self.assertEqual(len(out), 1)
        self.assertIn("trailing_pullback", out[0].triggers)

    def test_no_pullback_when_at_peak(self) -> None:
        pos = _pos(pid=1)
        snap = _OpenSnap.fresh(pnl_usd=10.0, pnl_pct=10.0, mfe=10.0)
        out = _reviewer(pullback_pct=50.0).evaluate(
            bot_owned_positions=[pos], symbol_for_id={1: "AAPL"},
            perf_open_states={1: snap},
        )
        self.assertEqual(out, [])

    def test_no_pullback_when_position_is_negative(self) -> None:
        # MFE never went positive → no trailing trigger.
        pos = _pos(pid=1)
        snap = _OpenSnap.fresh(pnl_usd=-3.0, pnl_pct=-3.0, mfe=0.0)
        out = _reviewer(pullback_pct=50.0, drawdown_pct=0.0).evaluate(
            bot_owned_positions=[pos], symbol_for_id={1: "AAPL"},
            perf_open_states={1: snap},
        )
        self.assertEqual(out, [])


class StaleHoldTriggerTests(unittest.TestCase):
    def test_flat_for_long_time_fires(self) -> None:
        pos = _pos(pid=1)
        snap = _OpenSnap.fresh(pnl_usd=0.1, pnl_pct=0.1, opened_ago_seconds=3700)
        out = _reviewer(stale_hold_minutes=60.0, stale_threshold_pct=0.5,
                        drawdown_pct=0.0, pullback_pct=0.0).evaluate(
            bot_owned_positions=[pos], symbol_for_id={1: "AAPL"},
            perf_open_states={1: snap},
        )
        self.assertEqual(len(out), 1)
        self.assertIn("stale_hold", out[0].triggers)

    def test_fresh_position_does_not_fire(self) -> None:
        pos = _pos(pid=1)
        snap = _OpenSnap.fresh(pnl_usd=0.0, pnl_pct=0.0, opened_ago_seconds=10)
        out = _reviewer().evaluate(
            bot_owned_positions=[pos], symbol_for_id={1: "AAPL"},
            perf_open_states={1: snap},
        )
        self.assertEqual(out, [])


class MaxHoldTriggerTests(unittest.TestCase):
    def test_held_past_ceiling(self) -> None:
        pos = _pos(pid=1)
        # Pos has +3% so drawdown doesn't fire; held 5h > 4h ceiling.
        snap = _OpenSnap.fresh(pnl_usd=3.0, pnl_pct=3.0, mfe=3.0,
                               opened_ago_seconds=5 * 3600)
        out = _reviewer(max_hold_minutes=240.0).evaluate(
            bot_owned_positions=[pos], symbol_for_id={1: "AAPL"},
            perf_open_states={1: snap},
        )
        triggers = out[0].triggers if out else []
        self.assertIn("max_hold", triggers)


class FallbackPathTests(unittest.TestCase):
    def test_uses_live_rate_when_no_open_state(self) -> None:
        # When the tracker hasn't observed the position yet, the reviewer
        # falls back to deriving pct from live rates + pos.open_rate.
        pos = _pos(pid=1, open_rate=100.0)

        class _Rate:
            def __init__(self, mid: float) -> None:
                self.mid = mid

        # Live mid 97 vs entry 100 → -3% on a long → drawdown trigger.
        out = _reviewer().evaluate(
            bot_owned_positions=[pos], symbol_for_id={1: "AAPL"},
            perf_open_states=None,
            live_rates={1: _Rate(97.0)},
        )
        self.assertEqual(len(out), 1)
        self.assertIn("drawdown", out[0].triggers)


class NoTriggersTests(unittest.TestCase):
    def test_disabled_thresholds_emit_nothing(self) -> None:
        pos = _pos(pid=1)
        snap = _OpenSnap.fresh(pnl_usd=-99.0, pnl_pct=-50.0, mfe=10.0,
                               opened_ago_seconds=99999)
        out = _reviewer(drawdown_pct=0.0, pullback_pct=0.0,
                        stale_hold_minutes=0.0, max_hold_minutes=0.0).evaluate(
            bot_owned_positions=[pos], symbol_for_id={1: "AAPL"},
            perf_open_states={1: snap},
        )
        self.assertEqual(out, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
