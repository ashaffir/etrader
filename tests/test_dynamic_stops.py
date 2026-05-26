"""Regression tests for :mod:`src.execution.dynamic_stops`.

The store is the single source of truth for per-position SL/TP and
trailing bands. Tests cover:

- read-through fallback to guardrail defaults
- override write/read round-trip
- trailing-stop ratcheting + floor semantics
- snapshot ↔ restore (used by the persistence layer)
"""

from __future__ import annotations

import unittest

from src.execution.dynamic_stops import DynamicStopsStore, StopBand


def _store(sl: float = 5.0, tp: float = 8.0) -> DynamicStopsStore:
    return DynamicStopsStore(default_stop_loss_pct=sl, default_take_profit_pct=tp)


class FallbackTests(unittest.TestCase):
    def test_no_override_returns_default(self) -> None:
        s = _store(sl=4.0, tp=6.0)
        band = s.effective_band(position_id=42)
        self.assertEqual(band.stop_loss_pct, 4.0)
        self.assertEqual(band.take_profit_pct, 6.0)
        self.assertIsNone(band.trailing_stop_pct)
        self.assertFalse(s.has_override(42))


class SetBandTests(unittest.TestCase):
    def test_set_all_three_fields(self) -> None:
        s = _store()
        band = s.set_band(7, stop_loss_pct=2.5, take_profit_pct=5.0, trailing_stop_pct=1.0)
        self.assertEqual(band.stop_loss_pct, 2.5)
        self.assertEqual(band.take_profit_pct, 5.0)
        self.assertEqual(band.trailing_stop_pct, 1.0)
        self.assertTrue(s.has_override(7))

    def test_partial_update_keeps_other_fields(self) -> None:
        s = _store(sl=5.0, tp=8.0)
        s.set_band(7, stop_loss_pct=2.0, take_profit_pct=6.0)
        s.set_band(7, trailing_stop_pct=1.5)  # only set trailing
        band = s.effective_band(7)
        self.assertEqual(band.stop_loss_pct, 2.0)   # preserved
        self.assertEqual(band.take_profit_pct, 6.0)  # preserved
        self.assertEqual(band.trailing_stop_pct, 1.5)  # new

    def test_clear_removes_override(self) -> None:
        s = _store(sl=5.0, tp=8.0)
        s.set_band(7, stop_loss_pct=2.0)
        s.clear(7)
        self.assertFalse(s.has_override(7))
        band = s.effective_band(7)
        self.assertEqual(band.stop_loss_pct, 5.0)  # back to default


class TrailingTests(unittest.TestCase):
    def test_no_trailing_floor_is_none(self) -> None:
        s = _store()
        s.set_band(1, stop_loss_pct=3.0, take_profit_pct=6.0)
        self.assertIsNone(s.trailing_floor_pct(1))

    def test_ratchet_records_new_mfe(self) -> None:
        s = _store()
        s.set_band(1, stop_loss_pct=3.0, take_profit_pct=10.0, trailing_stop_pct=2.0)
        # First ratchet at +5%: MFE becomes 5, trailing floor = 5-2 = +3.
        s.ratchet_trailing(1, current_pnl_pct=5.0)
        self.assertEqual(s.trailing_floor_pct(1), 3.0)
        # Lower P/L doesn't move the floor down.
        s.ratchet_trailing(1, current_pnl_pct=3.5)
        self.assertEqual(s.trailing_floor_pct(1), 3.0)
        # Higher P/L moves the floor up.
        s.ratchet_trailing(1, current_pnl_pct=8.0)
        self.assertEqual(s.trailing_floor_pct(1), 6.0)

    def test_trailing_floor_never_loose_than_base(self) -> None:
        s = _store()
        # Position has SL=4% and trail=2%. Until MFE crosses, floor = -4%.
        s.set_band(1, stop_loss_pct=4.0, take_profit_pct=10.0, trailing_stop_pct=2.0)
        self.assertEqual(s.trailing_floor_pct(1), -4.0)
        # Even after a small positive excursion (MFE=1%), trailing floor
        # would be -1%, but we clamp to the base -4%.
        s.ratchet_trailing(1, current_pnl_pct=1.0)
        self.assertEqual(s.trailing_floor_pct(1), -4.0)


class PersistTests(unittest.TestCase):
    def test_snapshot_round_trip(self) -> None:
        s = _store()
        s.set_band(11, stop_loss_pct=2.0, take_profit_pct=6.0,
                   trailing_stop_pct=1.5, rationale="bank winner")
        s.set_band(22, stop_loss_pct=4.0, take_profit_pct=9.0)
        s.ratchet_trailing(11, current_pnl_pct=10.0)  # bump mfe_pct
        payload = s.to_persistable()
        s2 = _store()
        s2.restore(payload)
        b11 = s2.effective_band(11)
        b22 = s2.effective_band(22)
        self.assertEqual(b11.stop_loss_pct, 2.0)
        self.assertEqual(b11.take_profit_pct, 6.0)
        self.assertEqual(b11.trailing_stop_pct, 1.5)
        self.assertEqual(b11.mfe_pct, 10.0)
        self.assertEqual(b11.rationale, "bank winner")
        self.assertEqual(b22.stop_loss_pct, 4.0)
        self.assertEqual(b22.take_profit_pct, 9.0)
        self.assertIsNone(b22.trailing_stop_pct)

    def test_restore_skips_junk(self) -> None:
        s = _store()
        s.restore({"not_an_int": {"stop_loss_pct": 2.0}})
        s.restore({"7": "also not a dict"})  # type: ignore[arg-type]
        self.assertEqual(s.snapshot(), {})


class StopBandTests(unittest.TestCase):
    def test_dict_round_trip(self) -> None:
        b = StopBand(stop_loss_pct=2.5, take_profit_pct=5.0,
                     trailing_stop_pct=1.0, mfe_pct=3.5, rationale="x")
        b2 = StopBand.from_dict(b.to_dict())
        self.assertEqual(b, b2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
