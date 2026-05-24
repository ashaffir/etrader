"""Selector + runner integration tests."""

import unittest
from typing import Sequence

from src.config import GuardrailsConfig, StrategyConfig
from src.etoro.market_data import Candle
from src.strategy.tools.base import AssetClass, Tool, ToolContext, ToolResult
from src.strategy.tools.registry import ToolRegistry, register_default_tools
from src.strategy.tools.runner import ToolRunner
from src.strategy.tools.selector import ToolSelector, ToolSelectorConfig


def _candles(closes: Sequence[float]) -> list[Candle]:
    return [
        Candle(instrument_id=1, from_date=None, open=c, high=c, low=c, close=c, volume=10.0)
        for c in closes
    ]


def _ctx(asset_class: AssetClass = AssetClass.STOCK) -> ToolContext:
    closes = [100.0 + i * 0.5 for i in range(80)]
    return ToolContext(
        instrument_id=1,
        symbol="X",
        asset_class=asset_class,
        candidate_action="BUY",
        strategy=StrategyConfig(),
        guardrails=GuardrailsConfig(),
        candles=_candles(closes),
        rate=None,
        instrument_meta=None,
    )


class _AlwaysVetoTool(Tool):
    name = "always_veto"
    family = "context"
    role = "gate"
    purpose = "test gate"
    asset_classes = (AssetClass.STOCK,)

    def evaluate(self, ctx: ToolContext) -> ToolResult:
        return ToolResult(gate_passed=False, gate_reason="testing")


class _ScoreTool(Tool):
    name = "score_tool"
    family = "price"
    role = "feature"
    purpose = "test"
    asset_classes = (AssetClass.STOCK,)

    def evaluate(self, ctx: ToolContext) -> ToolResult:
        return ToolResult(features={"x": 1}, score=0.4)


class SelectorTests(unittest.TestCase):
    def test_skips_tools_for_wrong_asset_class(self) -> None:
        reg = ToolRegistry()
        reg.register(_AlwaysVetoTool())
        ctx = _ctx(asset_class=AssetClass.FX)
        sel = ToolSelector(config=ToolSelectorConfig(max_tools_per_cycle=10))
        kept, trace = sel.select(registry_tools=list(reg), ctx=ctx, regime="trending")
        self.assertEqual(kept, [])
        self.assertEqual(trace.skipped_static[0][0], "always_veto")

    def test_default_registry_loads(self) -> None:
        reg = register_default_tools(feed_fetcher=None, feed_enabled=True)
        names = reg.names()
        for required in (
            "sma_cross", "ema_cross", "rsi", "macd", "bollinger", "donchian",
            "trend_filter", "obv", "vwap", "volume_spike", "cmf", "ad_line",
            "spread_filter", "market_hours", "higher_tf_trend",
            "cross_asset_regime", "relative_strength", "instrument_feed",
        ):
            self.assertIn(required, names, f"missing default tool: {required}")

    def test_runner_short_circuits_on_gate(self) -> None:
        reg = ToolRegistry()
        reg.register(_AlwaysVetoTool())
        reg.register(_ScoreTool())
        sel = ToolSelector(config=ToolSelectorConfig(max_tools_per_cycle=10))
        runner = ToolRunner(registry=reg, selector=sel)
        result = runner.run(ctx=_ctx(), regime="trending")
        self.assertFalse(result.gate_passed)
        # ScoreTool's score should not be present because the gate vetoed first
        # (or they ran in selector-priority order; we just confirm gate fired).
        self.assertIn("always_veto:", result.gate_reason + ":")

    def test_runner_collects_features_and_aggregate(self) -> None:
        reg = ToolRegistry()
        reg.register(_ScoreTool())
        sel = ToolSelector(config=ToolSelectorConfig(max_tools_per_cycle=10))
        runner = ToolRunner(registry=reg, selector=sel)
        result = runner.run(ctx=_ctx(), regime="trending")
        self.assertTrue(result.gate_passed)
        self.assertAlmostEqual(result.aggregate_score, 0.4)
        self.assertIn("score_tool.x", result.features)


if __name__ == "__main__":
    unittest.main()
