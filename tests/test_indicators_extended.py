"""Tests for the new indicators added in the tools rollout."""

import unittest

from src.strategy.indicators import (
    accumulation_distribution_line,
    bollinger_bands,
    chaikin_money_flow,
    donchian_channel,
    macd,
    on_balance_volume,
    volume_spike_ratio,
    vwap,
)


class MacdTests(unittest.TestCase):
    def test_returns_three_aligned_series(self) -> None:
        values = list(range(1, 80))
        m, s, h = macd(values, fast=12, slow=26, signal=9)
        self.assertEqual(len(m), len(values))
        self.assertEqual(len(s), len(values))
        self.assertEqual(len(h), len(values))
        # In a strictly rising series, MACD, signal, and histogram are positive late.
        self.assertGreater(m[-1] or 0.0, 0.0)
        self.assertGreater(s[-1] or 0.0, 0.0)


class BollingerTests(unittest.TestCase):
    def test_constant_series_collapses_bands(self) -> None:
        values = [100.0] * 25
        lower, middle, upper = bollinger_bands(values, period=20, stddev=2.0)
        self.assertAlmostEqual(lower[-1], middle[-1])
        self.assertAlmostEqual(upper[-1], middle[-1])

    def test_volatile_series_widens_bands(self) -> None:
        values = [100.0 + (5.0 if i % 2 == 0 else -5.0) for i in range(40)]
        lower, middle, upper = bollinger_bands(values, period=20, stddev=2.0)
        self.assertGreater(upper[-1], middle[-1])
        self.assertLess(lower[-1], middle[-1])


class DonchianTests(unittest.TestCase):
    def test_breakout_at_top(self) -> None:
        highs = [1.0] * 19 + [100.0]
        lows = [0.5] * 19 + [99.0]
        lower, upper = donchian_channel(highs, lows, period=20)
        self.assertEqual(upper[-1], 100.0)
        self.assertEqual(lower[-1], 0.5)


class VolumeBasedTests(unittest.TestCase):
    def test_obv_rising_with_price(self) -> None:
        closes = [100, 101, 102, 103]
        volumes = [10, 10, 10, 10]
        obv = on_balance_volume(closes, volumes)
        self.assertEqual(obv[0], 0.0)
        self.assertEqual(obv[-1], 30.0)

    def test_vwap_returns_none_when_no_volume(self) -> None:
        v = vwap([1, 1], [1, 1], [1, 1], [0, 0])
        self.assertIsNone(v)

    def test_volume_spike_ratio_detects_spike(self) -> None:
        volumes = [10.0] * 20 + [50.0]
        ratio = volume_spike_ratio(volumes, lookback=20)
        self.assertIsNotNone(ratio)
        self.assertGreaterEqual(ratio, 4.5)

    def test_cmf_zero_when_no_volume(self) -> None:
        v = chaikin_money_flow(
            [2.0] * 20, [1.0] * 20, [1.5] * 20, [0.0] * 20, period=20,
        )
        self.assertIsNone(v)

    def test_ad_line_runs_for_normal_input(self) -> None:
        n = 20
        ad = accumulation_distribution_line(
            [2.0] * n, [1.0] * n, [1.5] * n, [10.0] * n,
        )
        self.assertEqual(len(ad), n)
        # 1.5 is exactly the midpoint, so each bar's MFM is 0 → flat A/D.
        self.assertAlmostEqual(ad[-1] or 0.0, 0.0)


if __name__ == "__main__":
    unittest.main()
