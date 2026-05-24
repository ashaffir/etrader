"""Tests for context tools — gates, regime hooks, relative strength."""

import unittest

from src.config import GuardrailsConfig, StrategyConfig
from src.etoro.market_data import Candle, LiveRate
from src.strategy.tools.base import AssetClass, ToolContext
from src.strategy.tools.context_tools import (
    CrossAssetRegimeTool,
    HigherTfTrendTool,
    MarketHoursTool,
    RelativeStrengthTool,
    SpreadFilterTool,
)
from src.strategy.tools.feed_tool import InstrumentFeedTool


def _candles(closes: list[float]) -> list[Candle]:
    return [
        Candle(instrument_id=1, from_date=None, open=c, high=c, low=c, close=c, volume=0.0)
        for c in closes
    ]


def _ctx(
    closes: list[float],
    *,
    action: str = "BUY",
    asset_class: AssetClass = AssetClass.STOCK,
    rate: LiveRate | None = None,
    higher_tf: list[float] | None = None,
    regime: dict | None = None,
) -> ToolContext:
    return ToolContext(
        instrument_id=1,
        symbol="AAPL",
        asset_class=asset_class,
        candidate_action=action,
        strategy=StrategyConfig(),
        guardrails=GuardrailsConfig(),
        candles=_candles(closes),
        rate=rate,
        instrument_meta=None,
        higher_tf_candles=_candles(higher_tf) if higher_tf else (),
        cross_asset_regime=regime,
    )


class SpreadFilterTests(unittest.TestCase):
    def test_wide_spread_vetoes_buy(self) -> None:
        rate = LiveRate(instrument_id=1, ask=101.0, bid=99.0, last=100.0, timestamp=None)
        result = SpreadFilterTool().evaluate(_ctx([100.0] * 10, rate=rate))
        self.assertFalse(result.gate_passed)
        self.assertIn("spread", result.gate_reason)

    def test_tight_spread_passes(self) -> None:
        rate = LiveRate(instrument_id=1, ask=100.05, bid=100.0, last=100.0, timestamp=None)
        result = SpreadFilterTool().evaluate(_ctx([100.0] * 10, rate=rate))
        self.assertTrue(result.gate_passed)

    def test_no_rate_does_not_gate(self) -> None:
        result = SpreadFilterTool().evaluate(_ctx([100.0] * 10, rate=None))
        self.assertTrue(result.gate_passed)


class MarketHoursTests(unittest.TestCase):
    def test_crypto_skipped_via_applies_to(self) -> None:
        ctx = _ctx([100.0] * 5, asset_class=AssetClass.CRYPTO)
        # CRYPTO is intentionally excluded from MarketHoursTool's asset_classes.
        self.assertFalse(MarketHoursTool().applies_to(ctx))


class HigherTfTrendTests(unittest.TestCase):
    def test_trend_unknown_when_too_few_bars(self) -> None:
        result = HigherTfTrendTool().evaluate(_ctx([100.0] * 60, higher_tf=[100.0] * 5))
        self.assertEqual(result.features["trend"], "unknown")

    def test_uptrend_supports_buy(self) -> None:
        rising = [100.0 + i * 0.5 for i in range(80)]
        result = HigherTfTrendTool().evaluate(
            _ctx([100.0] * 60, higher_tf=rising, action="BUY"),
        )
        self.assertEqual(result.features["trend"], "up")
        self.assertGreater(result.score, 0.0)


class CrossAssetRegimeTests(unittest.TestCase):
    def test_no_regime_yields_zero_score(self) -> None:
        result = CrossAssetRegimeTool().evaluate(_ctx([100.0] * 30, regime=None))
        self.assertEqual(result.score, 0.0)

    def test_risk_on_supports_buy(self) -> None:
        regime = {"risk_on": True, "spx_trend": "up", "btc_trend": "flat"}
        result = CrossAssetRegimeTool().evaluate(_ctx([100.0] * 30, regime=regime))
        self.assertGreater(result.score, 0.0)


class RelativeStrengthTests(unittest.TestCase):
    def test_outperformer_scores_positive(self) -> None:
        rising = [100.0 + i * 0.5 for i in range(40)]
        regime = {"spx_momentum_pct": 1.0}
        result = RelativeStrengthTool().evaluate(_ctx(rising, action="BUY", regime=regime))
        self.assertGreater(result.score, 0.0)


class InstrumentFeedToolTests(unittest.TestCase):
    def test_no_fetcher_yields_zero_score(self) -> None:
        tool = InstrumentFeedTool(fetcher=None)
        result = tool.evaluate(_ctx([100.0] * 5))
        self.assertEqual(result.score, 0.0)
        # applies_to returns False when fetcher is None
        self.assertFalse(tool.applies_to(_ctx([100.0] * 5)))

    def test_with_stub_fetcher(self) -> None:
        class _Stub:
            def fetch(self, _id):
                class _S:
                    post_count = 8
                    posts_24h = 3
                    bullish_keyword_count = 5
                    bearish_keyword_count = 1
                    sample_titles = ("nice breakout",)
                return _S()
        tool = InstrumentFeedTool(fetcher=_Stub())
        result = tool.evaluate(_ctx([100.0] * 5, action="BUY"))
        self.assertGreater(result.score, 0.0)
        self.assertEqual(result.features["bullish_keywords"], 5)


if __name__ == "__main__":
    unittest.main()
