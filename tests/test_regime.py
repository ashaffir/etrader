"""Tests for instrument + cross-asset regime detection."""

import unittest

from src.etoro.market_data import Candle
from src.strategy.regime import (
    detect_cross_asset_regime,
    detect_instrument_regime,
)


def _candles(closes: list[float], *, hl_pad: float = 0.5) -> list[Candle]:
    return [
        Candle(
            instrument_id=1, from_date=None, open=c,
            high=c + hl_pad, low=c - hl_pad, close=c, volume=0.0,
        )
        for c in closes
    ]


class InstrumentRegimeTests(unittest.TestCase):
    def test_trending_when_far_from_sma(self) -> None:
        # Hard upward drift: price will leave SMA50 by many ATRs.
        closes = [100.0 + i * 1.5 for i in range(80)]
        regime = detect_instrument_regime(_candles(closes))
        self.assertEqual(regime.label, "trending")

    def test_unknown_when_too_few_bars(self) -> None:
        regime = detect_instrument_regime(_candles([100.0] * 10))
        self.assertEqual(regime.label, "unknown")


class CrossAssetRegimeTests(unittest.TestCase):
    def test_risk_on_when_spx_up(self) -> None:
        rising = [100.0 + i * 0.4 for i in range(120)]
        flat = [100.0] * 120
        regime = detect_cross_asset_regime(
            spx_candles=_candles(rising),
            btc_candles=_candles(flat),
        )
        self.assertTrue(regime.risk_on)
        self.assertEqual(regime.spx_trend, "up")

    def test_risk_off_when_spx_down_btc_down(self) -> None:
        falling = [200.0 - i * 0.4 for i in range(120)]
        regime = detect_cross_asset_regime(
            spx_candles=_candles(falling),
            btc_candles=_candles(falling),
        )
        self.assertFalse(regime.risk_on)
        self.assertEqual(regime.spx_trend, "down")
        self.assertEqual(regime.btc_trend, "down")


if __name__ == "__main__":
    unittest.main()
