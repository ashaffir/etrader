"""Signal-builder tests for the price-tool ensemble.

The signal layer no longer requires a strict ``SMA cross AND
RSI < overbought AND momentum > 0`` hit — every price tool feeds a
weighted ensemble whose signed raw_score determines BUY / CLOSE
candidacy. These tests cover:

- BUY emerges when several components vote bullish (clean trend
  re-ignition).
- CLOSE emerges for an owned position when several components vote
  bearish (rally → reversal).
- A chronically over-extended uptrend with no fresh trigger does NOT
  qualify as a BUY at the production min_signal_strength threshold.
- ``min_signal_strength`` and ``min_exit_strength`` actually prune.
- Short history is skipped.
"""

from __future__ import annotations

import unittest
from typing import Sequence

from src.config import StrategyConfig
from src.etoro.market_data import Candle
from src.strategy.signals import build_candidates


def _candles_from_closes(instrument_id: int, closes: Sequence[float]) -> list[Candle]:
    """Build Candle objects with high == low == close, volume = 0."""
    return [
        Candle(
            instrument_id=instrument_id,
            from_date=None,
            open=c, high=c, low=c, close=c,
            volume=0.0,
        )
        for c in closes
    ]


def _bullish_recovery_closes() -> list[float]:
    """Long sideways base → shallow dip → re-ignition rally.

    Calibrated so several components fire bullish at once: fresh SMA
    and EMA bull crosses inside the 5-bar lookback, MACD histogram
    crosses zero, Donchian upper extends past prior high, and momentum
    is strongly positive. Some bearish votes (RSI overbought from the
    fast rally, Bollinger upper extension) push back, but the
    aggregate clears the production min_signal_strength bar.
    """
    sideways = [100.0] * 35
    dip = [98.5, 97.0, 95.5, 94.0]
    rally = [95.5, 98.5, 102.0, 105.5, 109.0, 112.5]
    return sideways + dip + rally


def _rally_then_dump_closes() -> list[float]:
    """Long slow climb → 17-bar steep reversal.

    Triggers Donchian breakdown, strongly negative momentum, MACD
    histogram below zero, and meaningful SMA/EMA spread inversion —
    enough to push the aggregate well into negative territory.
    """
    climb = [100.0 + i * 0.5 for i in range(45)]    # 100 → 122.5 over 45 bars
    peak_hold = [climb[-1]] * 2
    dump_arr = [climb[-1] - i * 4.0 for i in range(1, 18)]
    return climb + peak_hold + dump_arr


def _toppy_closes(n: int = 70) -> list[float]:
    """Linear +1/bar — over-extended, no fresh trigger."""
    return [100.0 + i for i in range(n)]


def _flat_closes(n: int = 60) -> list[float]:
    """Genuinely flat — every component must score zero."""
    return [100.0] * n


class SignalsTests(unittest.TestCase):
    def setUp(self) -> None:
        # Production-realistic config (defaults), unless a test overrides.
        self.cfg = StrategyConfig()

    def test_bullish_recovery_yields_buy(self) -> None:
        cands = build_candidates(
            cfg=self.cfg,
            candles_by_instrument={1: _candles_from_closes(1, _bullish_recovery_closes())},
            symbol_for_id={1: "AAPL"},
            bot_owned_instrument_ids=set(),
        )
        self.assertEqual(len(cands), 1, msg=f"expected 1 BUY, got {len(cands)}")
        c = cands[0]
        self.assertEqual(c.action, "BUY")
        self.assertEqual(c.symbol, "AAPL")
        self.assertGreaterEqual(c.strength, self.cfg.min_signal_strength)
        # Component breakdown must be exposed verbatim:
        names = {comp.name for comp in c.components}
        self.assertIn("sma_cross", names)
        self.assertIn("macd", names)
        self.assertIn("momentum", names)

    def test_rally_then_dump_yields_close_when_owned(self) -> None:
        cands = build_candidates(
            cfg=self.cfg,
            candles_by_instrument={2: _candles_from_closes(2, _rally_then_dump_closes())},
            symbol_for_id={2: "MSFT"},
            bot_owned_instrument_ids={2},
        )
        self.assertEqual(len(cands), 1, msg=f"expected 1 CLOSE, got {len(cands)}")
        c = cands[0]
        self.assertEqual(c.action, "CLOSE")
        self.assertGreaterEqual(c.strength, self.cfg.min_exit_strength)
        # raw_score must be negative — that's how a CLOSE arises in the new model.
        self.assertLess(c.raw_score, 0.0)

    def test_rally_then_dump_unowned_yields_no_buy(self) -> None:
        # Same series, but bot doesn't own → CLOSE is irrelevant, no BUY arises.
        cands = build_candidates(
            cfg=self.cfg,
            candles_by_instrument={2: _candles_from_closes(2, _rally_then_dump_closes())},
            symbol_for_id={2: "MSFT"},
            bot_owned_instrument_ids=set(),
        )
        self.assertEqual(cands, [])

    def test_toppy_does_not_qualify_as_buy(self) -> None:
        # Pure up-trend has mixed signals: RSI overbought + Bollinger upper extension
        # cancel Donchian/momentum, leaving the ensemble too weak to clear
        # min_signal_strength.
        cands = build_candidates(
            cfg=self.cfg,
            candles_by_instrument={3: _candles_from_closes(3, _toppy_closes())},
            symbol_for_id={3: "X"},
            bot_owned_instrument_ids=set(),
        )
        self.assertEqual(cands, [])

    def test_flat_series_yields_nothing(self) -> None:
        cands = build_candidates(
            cfg=self.cfg,
            candles_by_instrument={4: _candles_from_closes(4, _flat_closes())},
            symbol_for_id={4: "FLAT"},
            bot_owned_instrument_ids=set(),
        )
        self.assertEqual(cands, [])

    def test_min_strength_prunes(self) -> None:
        cfg = StrategyConfig(min_signal_strength=0.95)
        cands = build_candidates(
            cfg=cfg,
            candles_by_instrument={1: _candles_from_closes(1, _bullish_recovery_closes())},
            symbol_for_id={1: "AAPL"},
            bot_owned_instrument_ids=set(),
        )
        self.assertEqual(cands, [])

    def test_min_exit_strength_prunes(self) -> None:
        cfg = StrategyConfig(min_exit_strength=0.95)
        cands = build_candidates(
            cfg=cfg,
            candles_by_instrument={2: _candles_from_closes(2, _rally_then_dump_closes())},
            symbol_for_id={2: "MSFT"},
            bot_owned_instrument_ids={2},
        )
        self.assertEqual(cands, [])

    def test_short_history_skipped(self) -> None:
        cands = build_candidates(
            cfg=self.cfg,
            candles_by_instrument={1: _candles_from_closes(1, [100.0] * 5)},
            symbol_for_id={1: "AAPL"},
            bot_owned_instrument_ids=set(),
        )
        self.assertEqual(cands, [])

    def test_disabled_components_via_zero_weight(self) -> None:
        """Weight=0 must remove a component from the aggregate but keep it visible."""
        cfg = StrategyConfig(
            weight_donchian=0.0,
            weight_macd=0.0,
            weight_bollinger=0.0,
        )
        cands = build_candidates(
            cfg=cfg,
            candles_by_instrument={1: _candles_from_closes(1, _bullish_recovery_closes())},
            symbol_for_id={1: "AAPL"},
            bot_owned_instrument_ids=set(),
        )
        # Should still produce a BUY (SMA + EMA + RSI + momentum carry it).
        self.assertEqual(len(cands), 1)
        # The disabled components are still present in the report (zero weight, not
        # zero score) so the operator can see why they were ignored.
        names = {c.name for c in cands[0].components}
        self.assertIn("donchian", names)
        self.assertIn("macd", names)
        self.assertIn("bollinger", names)


if __name__ == "__main__":
    unittest.main()
