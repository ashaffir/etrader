"""Indicators unit tests."""

import unittest

from src.strategy.indicators import (
    average_true_range,
    exponential_moving_average,
    momentum_pct,
    relative_strength_index,
    simple_moving_average,
)


class SmaTests(unittest.TestCase):
    def test_returns_none_until_period_filled(self) -> None:
        self.assertEqual(
            simple_moving_average([1, 2, 3, 4], 3),
            [None, None, 2.0, 3.0],
        )

    def test_rejects_zero_period(self) -> None:
        with self.assertRaises(ValueError):
            simple_moving_average([1, 2, 3], 0)

    def test_constant_series(self) -> None:
        out = simple_moving_average([5, 5, 5, 5, 5], 3)
        self.assertEqual(out, [None, None, 5.0, 5.0, 5.0])


class EmaTests(unittest.TestCase):
    def test_seed_equals_sma_then_recursive(self) -> None:
        values = [1, 2, 3, 4, 5, 6]
        out = exponential_moving_average(values, 3)
        self.assertIsNone(out[0])
        self.assertIsNone(out[1])
        self.assertAlmostEqual(out[2], (1 + 2 + 3) / 3)
        # Subsequent values should track upward but be < the latest input.
        self.assertGreater(out[3], out[2])
        self.assertLess(out[5], values[-1])


class RsiTests(unittest.TestCase):
    def test_all_gains_returns_100(self) -> None:
        values = list(range(1, 30))  # strictly increasing
        out = relative_strength_index(values, 14)
        self.assertEqual(out[14], 100.0)

    def test_all_losses_returns_zero(self) -> None:
        values = list(range(30, 0, -1))
        out = relative_strength_index(values, 14)
        self.assertEqual(out[14], 0.0)

    def test_insufficient_data_returns_all_none(self) -> None:
        out = relative_strength_index([1, 2, 3], 14)
        self.assertTrue(all(v is None for v in out))


class MomentumTests(unittest.TestCase):
    def test_basic_lookback(self) -> None:
        # +10% over 1 lookback step
        self.assertEqual(momentum_pct([100, 110], 1), 10.0)

    def test_returns_none_when_not_enough_data(self) -> None:
        self.assertIsNone(momentum_pct([100], 1))

    def test_zero_division_safe(self) -> None:
        self.assertIsNone(momentum_pct([0, 5], 1))


class AtrTests(unittest.TestCase):
    def test_minimum_data_required(self) -> None:
        self.assertIsNone(average_true_range([1], [1], [1], 14))

    def test_simple_atr(self) -> None:
        # All TRs == 1; ATR should also be 1.
        n = 30
        highs = [10.0 + i * 0 + 0.5 for i in range(n)]
        lows = [10.0 + i * 0 - 0.5 for i in range(n)]
        closes = [10.0 + i * 0 for i in range(n)]
        atr = average_true_range(highs, lows, closes, 14)
        self.assertIsNotNone(atr)
        self.assertAlmostEqual(atr, 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
