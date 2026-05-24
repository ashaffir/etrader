"""Unit tests for the price-tool ensemble component scorers.

Each scorer must:
- Emit a signed score in [-1, +1].
- Behave monotonically with the underlying signal (more bullish input
  → more positive score).
- Return 0.0 with a clear ``insufficient data`` detail when the input
  doesn't yet have enough history for the indicator's lookback.
"""

from __future__ import annotations

import unittest

from src.config import StrategyConfig
from src.strategy.ensemble import (
    EnsembleResult,
    evaluate_ensemble,
)


class EnsembleAggregateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = StrategyConfig()

    def test_flat_series_produces_neutral_raw_score(self) -> None:
        closes = [100.0] * 60
        result = evaluate_ensemble(
            closes=closes, highs=closes, lows=closes, cfg=self.cfg,
        )
        self.assertIsInstance(result, EnsembleResult)
        self.assertAlmostEqual(result.raw_score, 0.0, places=2)
        self.assertEqual(result.buy_strength, 0.0)
        self.assertEqual(result.sell_strength, 0.0)

    def test_bullish_breakout_pushes_raw_score_positive(self) -> None:
        # Long flat region then a clean breakout — multiple components must vote up.
        closes = [100.0] * 50 + [101.0, 103.0, 105.0, 107.0, 109.0, 111.0, 113.0]
        result = evaluate_ensemble(
            closes=closes, highs=closes, lows=closes, cfg=self.cfg,
        )
        self.assertGreater(result.raw_score, 0.2,
                           msg=f"raw_score={result.raw_score} too weak for clean breakout")
        self.assertGreater(result.buy_strength, 0.2)
        self.assertEqual(result.sell_strength, 0.0)

    def test_bearish_reversal_pushes_raw_score_negative(self) -> None:
        # Long flat region then sharp drop — multiple components must vote down.
        closes = [100.0] * 50 + [99.0, 97.0, 95.0, 93.0, 91.0, 89.0, 87.0]
        result = evaluate_ensemble(
            closes=closes, highs=closes, lows=closes, cfg=self.cfg,
        )
        self.assertLess(result.raw_score, -0.2,
                        msg=f"raw_score={result.raw_score} too weak for clean reversal")
        self.assertGreater(result.sell_strength, 0.2)
        self.assertEqual(result.buy_strength, 0.0)

    def test_top_contributors_orders_by_absolute_score(self) -> None:
        closes = [100.0] * 50 + [101.0, 103.0, 105.0, 107.0, 109.0, 111.0, 113.0]
        result = evaluate_ensemble(
            closes=closes, highs=closes, lows=closes, cfg=self.cfg,
        )
        top = result.top_contributors(k=3)
        self.assertEqual(len(top), 3)
        # Top contributors must have non-decreasing |score|.
        absolutes = [abs(c.score) for c in top]
        self.assertEqual(absolutes, sorted(absolutes, reverse=True))

    def test_zero_weight_skips_component_from_aggregate(self) -> None:
        cfg_with = StrategyConfig()
        cfg_without = StrategyConfig(weight_macd=0.0)
        closes = [100.0] * 50 + [101.0, 103.0, 105.0, 107.0, 109.0, 111.0, 113.0]
        with_macd = evaluate_ensemble(
            closes=closes, highs=closes, lows=closes, cfg=cfg_with,
        )
        without_macd = evaluate_ensemble(
            closes=closes, highs=closes, lows=closes, cfg=cfg_without,
        )
        # Same components reported (zero-weight just drops it from the aggregate),
        # different total weight.
        self.assertEqual(
            sorted(c.name for c in with_macd.components),
            sorted(c.name for c in without_macd.components),
        )
        self.assertGreater(with_macd.total_weight, without_macd.total_weight)

    def test_insufficient_history_yields_safe_zeros(self) -> None:
        closes = [100.0] * 8  # nowhere near the 35-bar minimum
        result = evaluate_ensemble(
            closes=closes, highs=closes, lows=closes, cfg=self.cfg,
        )
        # Every component must report insufficient data with score == 0
        for comp in result.components:
            self.assertEqual(comp.score, 0.0,
                             msg=f"{comp.name} gave non-zero on insufficient data")

    def test_components_always_in_canonical_order(self) -> None:
        closes = [100.0] * 60
        result = evaluate_ensemble(
            closes=closes, highs=closes, lows=closes, cfg=self.cfg,
        )
        names = [c.name for c in result.components]
        self.assertEqual(
            names,
            ["sma_cross", "ema_cross", "rsi", "macd", "bollinger", "donchian", "momentum"],
        )


class ComponentBoundsTests(unittest.TestCase):
    """Every component must produce a score within [-1, +1]."""

    def setUp(self) -> None:
        self.cfg = StrategyConfig()

    def _check_bounds(self, closes: list[float]) -> None:
        result = evaluate_ensemble(
            closes=closes, highs=closes, lows=closes, cfg=self.cfg,
        )
        for c in result.components:
            self.assertGreaterEqual(c.score, -1.0,
                                    msg=f"{c.name}={c.score} below -1")
            self.assertLessEqual(c.score, 1.0,
                                 msg=f"{c.name}={c.score} above +1")

    def test_bounds_hold_for_breakout(self) -> None:
        self._check_bounds([100.0] * 50 + [110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0])

    def test_bounds_hold_for_breakdown(self) -> None:
        self._check_bounds([100.0] * 50 + [90.0, 85.0, 80.0, 75.0, 70.0, 65.0, 60.0])

    def test_bounds_hold_for_extreme_uptrend(self) -> None:
        self._check_bounds([100.0 + i * 5 for i in range(80)])


if __name__ == "__main__":
    unittest.main()
