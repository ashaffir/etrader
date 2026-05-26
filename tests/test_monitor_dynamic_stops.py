"""Tests for the monitor's use of per-position dynamic SL/TP bands.

Verifies that :meth:`PositionMonitor.positions_needing_close`:

- Honors the static guardrail defaults when no override is set.
- Uses the per-position override when present.
- Closes on the trailing-stop floor once MFE crosses the trailing band.
- Closes on the static SL even when trailing hasn't activated yet.
"""

from __future__ import annotations

import unittest

from src.etoro.market_data import LiveRate
from src.etoro.trading import Position
from src.execution.dynamic_stops import DynamicStopsStore
from src.execution.monitor import PositionMonitor


def _pos(pid: int = 1, iid: int = 1, open_rate: float = 100.0,
         is_buy: bool = True) -> Position:
    return Position(
        position_id=pid, instrument_id=iid, is_buy=is_buy,
        open_rate=open_rate, amount=100.0, units=1.0,
        leverage=1, mirror_id=0, pnl=0.0, raw={},
    )


def _rate(mid: float) -> LiveRate:
    # ``mid`` is derived from bid/ask in LiveRate; making them symmetric
    # around the desired mid keeps the math simple.
    return LiveRate(
        instrument_id=1, ask=mid + 0.01, bid=mid - 0.01,
        last=mid, timestamp="2026-01-01T00:00:00Z",
    )


class StaticDefaultsTests(unittest.TestCase):
    def test_below_default_sl_closes(self) -> None:
        m = PositionMonitor()
        pos = _pos(open_rate=100.0)
        # -6% drop, default SL=5%.
        out = m.positions_needing_close(
            bot_owned=[pos],
            rates={1: _rate(94.0)},
            stop_loss_pct=5.0, take_profit_pct=8.0,
        )
        self.assertEqual(len(out), 1)

    def test_within_band_no_close(self) -> None:
        m = PositionMonitor()
        pos = _pos(open_rate=100.0)
        # -3% drop, SL=5%.
        out = m.positions_needing_close(
            bot_owned=[pos], rates={1: _rate(97.0)},
            stop_loss_pct=5.0, take_profit_pct=8.0,
        )
        self.assertEqual(out, [])


class DynamicOverrideTests(unittest.TestCase):
    def test_per_position_tighter_sl_overrides_default(self) -> None:
        store = DynamicStopsStore(
            default_stop_loss_pct=5.0, default_take_profit_pct=8.0,
        )
        store.set_band(1, stop_loss_pct=2.0, take_profit_pct=5.0)
        m = PositionMonitor(dynamic_stops=store)
        pos = _pos(open_rate=100.0)
        # -3% drop. Default SL=5% wouldn't fire; per-position SL=2% does.
        out = m.positions_needing_close(
            bot_owned=[pos], rates={1: _rate(97.0)},
            stop_loss_pct=5.0, take_profit_pct=8.0,
        )
        self.assertEqual(len(out), 1)

    def test_trailing_floor_closes_after_ratchet(self) -> None:
        store = DynamicStopsStore(
            default_stop_loss_pct=5.0, default_take_profit_pct=20.0,
        )
        store.set_band(1, stop_loss_pct=5.0, take_profit_pct=20.0,
                       trailing_stop_pct=2.0)
        m = PositionMonitor(dynamic_stops=store)
        pos = _pos(open_rate=100.0)
        # First cycle: position is +6% (mid=106). Trailing activates
        # (6 > 2). Floor = 6 - 2 = +4%. No breach yet.
        out = m.positions_needing_close(
            bot_owned=[pos], rates={1: _rate(106.0)},
            stop_loss_pct=5.0, take_profit_pct=20.0,
        )
        self.assertEqual(out, [])
        # Second cycle: position pulled back to +3% (mid=103). Floor
        # remains at +4 (MFE peak was 6). +3 ≤ +4 → CLOSE.
        out = m.positions_needing_close(
            bot_owned=[pos], rates={1: _rate(103.0)},
            stop_loss_pct=5.0, take_profit_pct=20.0,
        )
        self.assertEqual(len(out), 1)

    def test_trailing_inactive_until_mfe_crosses(self) -> None:
        store = DynamicStopsStore(
            default_stop_loss_pct=5.0, default_take_profit_pct=20.0,
        )
        store.set_band(1, stop_loss_pct=4.0, take_profit_pct=20.0,
                       trailing_stop_pct=3.0)
        m = PositionMonitor(dynamic_stops=store)
        pos = _pos(open_rate=100.0)
        # +2% then -3.5%: trailing never activated (need > +3 to switch on),
        # so we use the static SL=4%. -3.5% does NOT breach -4%.
        m.positions_needing_close(
            bot_owned=[pos], rates={1: _rate(102.0)},
            stop_loss_pct=4.0, take_profit_pct=20.0,
        )
        out = m.positions_needing_close(
            bot_owned=[pos], rates={1: _rate(96.5)},
            stop_loss_pct=4.0, take_profit_pct=20.0,
        )
        self.assertEqual(out, [])
        # Now drop to -4.5% — breaches the static SL.
        out = m.positions_needing_close(
            bot_owned=[pos], rates={1: _rate(95.5)},
            stop_loss_pct=4.0, take_profit_pct=20.0,
        )
        self.assertEqual(len(out), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
