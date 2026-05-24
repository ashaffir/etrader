"""RSI + MACD oscillator tools — feature-only."""

from __future__ import annotations

from typing import Any

from ...indicators import macd, relative_strength_index
from ..base import AssetClass, Tool, ToolContext, ToolResult
from .moving_averages import _crossed_recently


_PRICE_ASSET_CLASSES = (
    AssetClass.STOCK,
    AssetClass.CRYPTO,
    AssetClass.ETF,
    AssetClass.INDEX,
    AssetClass.COMMODITY,
    AssetClass.FX,
    AssetClass.OTHER,
)


def _round(v: Any, ndigits: int = 4) -> Any:
    if v is None:
        return None
    try:
        return round(float(v), ndigits)
    except (TypeError, ValueError):
        return v


class RsiTool(Tool):
    name = "rsi"
    family = "price"
    role = "feature"
    purpose = "Wilder's RSI — overbought/oversold detection"
    asset_classes = _PRICE_ASSET_CLASSES

    def evaluate(self, ctx: ToolContext) -> ToolResult:
        closes = ctx.closes()
        rsi_series = relative_strength_index(closes, ctx.strategy.rsi_period)
        rsi = rsi_series[-1] if rsi_series else None
        zone = "neutral"
        score = 0.0
        if rsi is not None:
            if rsi >= ctx.strategy.rsi_overbought:
                zone = "overbought"
                score = 0.6 if ctx.candidate_action == "CLOSE" else -0.6
            elif rsi <= ctx.strategy.rsi_oversold:
                zone = "oversold"
                score = 0.6 if ctx.candidate_action == "BUY" else -0.3
            else:
                score = 0.2 if ctx.candidate_action == "BUY" and rsi < 60 else -0.1
        return ToolResult(features={"rsi": _round(rsi), "zone": zone}, score=score)


class MacdTool(Tool):
    name = "macd"
    family = "price"
    role = "feature"
    purpose = "MACD(12,26,9) — momentum-of-momentum trigger"
    asset_classes = _PRICE_ASSET_CLASSES

    def evaluate(self, ctx: ToolContext) -> ToolResult:
        closes = ctx.closes()
        macd_line, signal_line, histogram = macd(closes)
        m = macd_line[-1] if macd_line else None
        s = signal_line[-1] if signal_line else None
        h = histogram[-1] if histogram else None
        bullish_cross = _crossed_recently(macd_line, signal_line, up=True)
        bearish_cross = _crossed_recently(macd_line, signal_line, up=False)
        score = 0.0
        if bullish_cross and ctx.candidate_action == "BUY":
            score = 0.5
        elif bearish_cross and ctx.candidate_action == "CLOSE":
            score = 0.5
        elif h is not None:
            score = max(-0.3, min(0.3, h / max(abs(m or 1.0), 1e-6)))
            if ctx.candidate_action == "CLOSE":
                score = -score
        return ToolResult(
            features={
                "macd": _round(m),
                "signal": _round(s),
                "histogram": _round(h),
                "bullish_cross": bullish_cross,
                "bearish_cross": bearish_cross,
            },
            score=score,
        )
