"""Volume-tool tests, including the volume-availability auto-skip."""

import unittest

from src.config import GuardrailsConfig, StrategyConfig
from src.etoro.market_data import Candle
from src.strategy.tools.base import AssetClass, ToolContext
from src.strategy.tools.volume_tools import (
    AdLineTool,
    CmfTool,
    ObvTool,
    VolumeSpikeTool,
    VwapTool,
)


def _ctx_with_volume(volumes: list[float]) -> ToolContext:
    closes = [100.0 + i for i in range(len(volumes))]
    candles = [
        Candle(instrument_id=1, from_date=None, open=c, high=c + 0.5, low=c - 0.5, close=c, volume=v)
        for c, v in zip(closes, volumes)
    ]
    return ToolContext(
        instrument_id=1,
        symbol="AAPL",
        asset_class=AssetClass.STOCK,
        candidate_action="BUY",
        strategy=StrategyConfig(),
        guardrails=GuardrailsConfig(),
        candles=candles,
        rate=None,
        instrument_meta=None,
    )


class ApplicabilityTests(unittest.TestCase):
    def test_skips_when_no_volume(self) -> None:
        ctx = _ctx_with_volume([0.0] * 25)
        for tool in (ObvTool(), VwapTool(), VolumeSpikeTool(), CmfTool(), AdLineTool()):
            with self.subTest(tool=tool.name):
                self.assertFalse(tool.applies_to(ctx))

    def test_runs_when_volume_present(self) -> None:
        ctx = _ctx_with_volume([10.0] * 25)
        for tool in (ObvTool(), VwapTool(), VolumeSpikeTool(), CmfTool(), AdLineTool()):
            with self.subTest(tool=tool.name):
                self.assertTrue(tool.applies_to(ctx))


class ObvVwapTests(unittest.TestCase):
    def test_obv_emits_slope(self) -> None:
        ctx = _ctx_with_volume([10.0] * 25)
        result = ObvTool().evaluate(ctx)
        self.assertIn("slope_pct", result.features)

    def test_vwap_above_for_rising_series(self) -> None:
        # 25 rising bars with constant volume → last close > VWAP.
        ctx = _ctx_with_volume([10.0] * 25)
        result = VwapTool().evaluate(ctx)
        self.assertTrue(result.features.get("above_vwap"))


class VolumeSpikeTests(unittest.TestCase):
    def test_detects_5x_spike(self) -> None:
        vols = [10.0] * 20 + [50.0]
        ctx = _ctx_with_volume(vols)
        result = VolumeSpikeTool().evaluate(ctx)
        self.assertGreater(result.features["ratio"], 4.0)
        self.assertGreater(result.score, 0.0)


if __name__ == "__main__":
    unittest.main()
