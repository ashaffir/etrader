"""Smoke + behavior tests for price-family tools."""

import unittest
from typing import Sequence

from src.config import GuardrailsConfig, StrategyConfig
from src.etoro.market_data import Candle
from src.strategy.tools.base import AssetClass, ToolContext
from src.strategy.tools.price_tools import (
    BollingerTool,
    DonchianTool,
    EmaCrossTool,
    MacdTool,
    RsiTool,
    SmaCrossTool,
    TrendFilterTool,
)


def _candles(closes: Sequence[float], *, highs=None, lows=None) -> list[Candle]:
    h = highs or closes
    l = lows or closes
    return [
        Candle(instrument_id=1, from_date=None, open=c, high=hh, low=ll, close=c, volume=0.0)
        for c, hh, ll in zip(closes, h, l)
    ]


def _ctx(closes: Sequence[float], *, action: str = "BUY", highs=None, lows=None) -> ToolContext:
    return ToolContext(
        instrument_id=1,
        symbol="AAPL",
        asset_class=AssetClass.STOCK,
        candidate_action=action,
        strategy=StrategyConfig(min_signal_strength=0.0),
        guardrails=GuardrailsConfig(),
        candles=_candles(closes, highs=highs, lows=lows),
        rate=None,
        instrument_meta=None,
    )


def _rising(n: int = 80) -> list[float]:
    return [100.0 + i * 0.5 for i in range(n)]


def _bullish_cross() -> list[float]:
    sawtooth = [99.0 if i % 2 == 0 else 101.0 for i in range(55)]
    rising = [102.0, 103.0, 104.0, 105.0, 106.0]
    return sawtooth + rising


class SmaCrossToolTests(unittest.TestCase):
    def test_bull_cross_yields_positive_score(self) -> None:
        result = SmaCrossTool().evaluate(_ctx(_bullish_cross(), action="BUY"))
        self.assertTrue(result.features["crossed_up"])
        self.assertGreater(result.score, 0.0)


class EmaCrossToolTests(unittest.TestCase):
    def test_rising_series_has_features(self) -> None:
        result = EmaCrossTool().evaluate(_ctx(_rising(), action="BUY"))
        self.assertIn("ema_fast", result.features)


class RsiToolTests(unittest.TestCase):
    def test_overbought_close_scores_positive(self) -> None:
        ctx = _ctx(_rising(40), action="CLOSE")
        result = RsiTool().evaluate(ctx)
        # rising series → overbought → CLOSE score positive
        self.assertGreater(result.score, 0.0)
        self.assertEqual(result.features["zone"], "overbought")


class MacdToolTests(unittest.TestCase):
    def test_returns_macd_fields(self) -> None:
        result = MacdTool().evaluate(_ctx(_rising(), action="BUY"))
        self.assertIn("macd", result.features)
        self.assertIn("signal", result.features)


class BollingerToolTests(unittest.TestCase):
    def test_position_within_band(self) -> None:
        result = BollingerTool().evaluate(_ctx(_rising(40)))
        self.assertIn("position", result.features)


class DonchianToolTests(unittest.TestCase):
    def test_high_breakout(self) -> None:
        # 19 flat candles then a hard up-spike → breaks the 20-bar Donchian high.
        closes = [10.0] * 19 + [50.0]
        ctx = _ctx(closes, action="BUY", highs=closes, lows=[10.0] * 20)
        result = DonchianTool().evaluate(ctx)
        self.assertTrue(result.features["broke_high"])
        self.assertGreater(result.score, 0.0)


class TrendFilterToolTests(unittest.TestCase):
    def test_sharp_downtrend_vetoes_buy(self) -> None:
        closes = [200.0 - i * 0.6 for i in range(80)]
        ctx = _ctx(closes, action="BUY")
        result = TrendFilterTool().evaluate(ctx)
        self.assertFalse(result.gate_passed)
        self.assertIn("SMA50", result.gate_reason)

    def test_uptrend_does_not_veto(self) -> None:
        result = TrendFilterTool().evaluate(_ctx(_rising(), action="BUY"))
        self.assertTrue(result.gate_passed)


if __name__ == "__main__":
    unittest.main()
