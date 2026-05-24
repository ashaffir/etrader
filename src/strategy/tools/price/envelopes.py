"""Volatility envelopes (Bollinger, Donchian) + the trend filter gate."""

from __future__ import annotations

from typing import Any

from ...indicators import (
    bollinger_bands,
    donchian_channel,
    simple_moving_average,
)
from ..base import AssetClass, Tool, ToolContext, ToolResult


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


class BollingerTool(Tool):
    name = "bollinger"
    family = "price"
    role = "feature"
    purpose = "Bollinger position — % of band width from middle"
    asset_classes = _PRICE_ASSET_CLASSES

    def evaluate(self, ctx: ToolContext) -> ToolResult:
        closes = ctx.closes()
        lower, middle, upper = bollinger_bands(closes, period=20, stddev=2.0)
        m = middle[-1] if middle else None
        l = lower[-1] if lower else None
        u = upper[-1] if upper else None
        last_close = closes[-1] if closes else None
        position: float | None = None
        if l is not None and u is not None and u != l and last_close is not None:
            position = (last_close - l) / (u - l)
        score = 0.0
        if position is not None:
            if position <= 0.1 and ctx.candidate_action == "BUY":
                score = 0.5
            elif position >= 0.9 and ctx.candidate_action == "CLOSE":
                score = 0.5
            elif position >= 0.9 and ctx.candidate_action == "BUY":
                score = -0.4
        return ToolResult(
            features={
                "lower": _round(l),
                "middle": _round(m),
                "upper": _round(u),
                "position": _round(position, 3),
            },
            score=score,
        )


class DonchianTool(Tool):
    name = "donchian"
    family = "price"
    role = "feature"
    purpose = "Donchian breakout — new high/low over the last 20 bars"
    asset_classes = _PRICE_ASSET_CLASSES

    def evaluate(self, ctx: ToolContext) -> ToolResult:
        highs = ctx.highs()
        lows = ctx.lows()
        lower, upper = donchian_channel(highs, lows, period=20)
        l = lower[-1] if lower else None
        u = upper[-1] if upper else None
        closes = ctx.closes()
        last = closes[-1] if closes else None
        broke_high = last is not None and u is not None and last >= u
        broke_low = last is not None and l is not None and last <= l
        score = 0.0
        if broke_high and ctx.candidate_action == "BUY":
            score = 0.5
        elif broke_low and ctx.candidate_action == "CLOSE":
            score = 0.5
        elif broke_low and ctx.candidate_action == "BUY":
            score = -0.4
        return ToolResult(
            features={
                "lower": _round(l),
                "upper": _round(u),
                "broke_high": broke_high,
                "broke_low": broke_low,
            },
            score=score,
        )


class TrendFilterTool(Tool):
    """Gate that vetoes BUYs against the prevailing 50-bar trend."""

    name = "trend_filter"
    family = "price"
    role = "both"
    purpose = "Veto BUY when SMA50 slope is decisively negative"
    asset_classes = _PRICE_ASSET_CLASSES

    def evaluate(self, ctx: ToolContext) -> ToolResult:
        closes = ctx.closes()
        if len(closes) < 60:
            return ToolResult(features={"trend": "unknown"}, score=0.0)
        sma = simple_moving_average(closes, 50)
        sma_now = sma[-1]
        sma_then = sma[-11] if len(sma) >= 11 else None
        if sma_now is None or sma_then is None or sma_then <= 0:
            return ToolResult(features={"trend": "unknown"}, score=0.0)
        slope_pct = (sma_now - sma_then) / sma_then * 100.0
        if slope_pct < -1.5 and ctx.candidate_action == "BUY":
            return ToolResult(
                features={"trend": "down", "sma50_slope_pct": _round(slope_pct, 3)},
                score=-0.7,
                gate_passed=False,
                gate_reason=f"SMA50 slope {slope_pct:+.2f}% (< -1.5%)",
            )
        trend = "up" if slope_pct > 0.5 else ("down" if slope_pct < -0.5 else "flat")
        score = 0.0
        if trend == "up" and ctx.candidate_action == "BUY":
            score = 0.4
        elif trend == "down" and ctx.candidate_action == "CLOSE":
            score = 0.4
        return ToolResult(
            features={"trend": trend, "sma50_slope_pct": _round(slope_pct, 3)},
            score=score,
        )
