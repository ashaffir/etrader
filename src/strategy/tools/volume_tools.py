"""Volume-based tools: OBV, VWAP, volume_spike, CMF, A/D line.

All tools set ``requires_volume = True`` so the selector skips them
on instruments where eToro returns 0 volume (FX, indices). The
tools also check inline because asset-class metadata can be wrong.
"""

from __future__ import annotations

from typing import Any

from ..indicators import (
    accumulation_distribution_line,
    chaikin_money_flow,
    on_balance_volume,
    volume_spike_ratio,
    vwap,
)
from .base import AssetClass, Tool, ToolContext, ToolResult


_VOLUME_ASSET_CLASSES = (
    AssetClass.STOCK,
    AssetClass.CRYPTO,
    AssetClass.ETF,
    AssetClass.COMMODITY,
)


class ObvTool(Tool):
    name = "obv"
    family = "volume"
    role = "feature"
    purpose = "On-Balance Volume slope — confirms or warns vs price trend"
    asset_classes = _VOLUME_ASSET_CLASSES
    requires_volume = True

    def evaluate(self, ctx: ToolContext) -> ToolResult:
        obv = on_balance_volume(ctx.closes(), ctx.volumes())
        if len(obv) < 11:
            return ToolResult(features={"slope_pct": None}, score=0.0)
        a, b = obv[-11], obv[-1]
        if a in (None, 0) or b is None:
            return ToolResult(features={"slope_pct": None}, score=0.0)
        slope_pct = (b - a) / abs(a) * 100.0 if a else 0.0
        score = 0.0
        if ctx.candidate_action == "BUY":
            score = max(-0.4, min(0.4, slope_pct / 5.0))
        else:
            score = max(-0.4, min(0.4, -slope_pct / 5.0))
        return ToolResult(features={"slope_pct": _round(slope_pct, 3)}, score=score)


class VwapTool(Tool):
    name = "vwap"
    family = "volume"
    role = "feature"
    purpose = "VWAP — last close vs volume-weighted average price"
    asset_classes = _VOLUME_ASSET_CLASSES
    requires_volume = True

    def evaluate(self, ctx: ToolContext) -> ToolResult:
        v = vwap(ctx.highs(), ctx.lows(), ctx.closes(), ctx.volumes())
        closes = ctx.closes()
        last_close = closes[-1] if closes else None
        if v is None or last_close is None:
            return ToolResult(features={"vwap": None, "above_vwap": None}, score=0.0)
        above = last_close > v
        gap_pct = (last_close - v) / v * 100.0 if v > 0 else 0.0
        score = 0.0
        if ctx.candidate_action == "BUY":
            score = 0.3 if above else -0.2
        else:
            score = 0.3 if not above else -0.2
        return ToolResult(
            features={"vwap": _round(v), "above_vwap": above, "gap_pct": _round(gap_pct, 3)},
            score=score,
        )


class VolumeSpikeTool(Tool):
    name = "volume_spike"
    family = "volume"
    role = "feature"
    purpose = "Latest-bar volume vs 20-bar average — confirms breakouts"
    asset_classes = _VOLUME_ASSET_CLASSES
    requires_volume = True

    def evaluate(self, ctx: ToolContext) -> ToolResult:
        ratio = volume_spike_ratio(ctx.volumes(), lookback=20)
        if ratio is None:
            return ToolResult(features={"ratio": None}, score=0.0)
        score = 0.0
        if ratio >= 2.0 and ctx.candidate_action == "BUY":
            score = 0.4
        elif ratio < 0.6 and ctx.candidate_action == "BUY":
            score = -0.2
        elif ratio >= 2.0 and ctx.candidate_action == "CLOSE":
            score = 0.3
        return ToolResult(features={"ratio": _round(ratio, 3)}, score=score)


class CmfTool(Tool):
    name = "cmf"
    family = "volume"
    role = "feature"
    purpose = "Chaikin Money Flow — accumulation vs distribution over 20 bars"
    asset_classes = _VOLUME_ASSET_CLASSES
    requires_volume = True

    def evaluate(self, ctx: ToolContext) -> ToolResult:
        v = chaikin_money_flow(
            ctx.highs(), ctx.lows(), ctx.closes(), ctx.volumes(), period=20,
        )
        if v is None:
            return ToolResult(features={"cmf": None}, score=0.0)
        score = 0.0
        if ctx.candidate_action == "BUY":
            score = max(-0.4, min(0.4, v))
        else:
            score = max(-0.4, min(0.4, -v))
        return ToolResult(features={"cmf": _round(v, 4)}, score=score)


class AdLineTool(Tool):
    name = "ad_line"
    family = "volume"
    role = "feature"
    purpose = "Accumulation/Distribution line — slope confirms trend health"
    asset_classes = _VOLUME_ASSET_CLASSES
    requires_volume = True

    def evaluate(self, ctx: ToolContext) -> ToolResult:
        ad = accumulation_distribution_line(
            ctx.highs(), ctx.lows(), ctx.closes(), ctx.volumes(),
        )
        if len(ad) < 11:
            return ToolResult(features={"slope": None}, score=0.0)
        a, b = ad[-11], ad[-1]
        if a is None or b is None:
            return ToolResult(features={"slope": None}, score=0.0)
        slope = b - a
        # normalize roughly: divide by max|ad| over window so score
        # stays bounded across instruments at very different scales.
        scale = max(abs(x) for x in ad[-11:] if x is not None) or 1.0
        norm = slope / scale
        score = 0.0
        if ctx.candidate_action == "BUY":
            score = max(-0.4, min(0.4, norm))
        else:
            score = max(-0.4, min(0.4, -norm))
        return ToolResult(features={"slope": _round(slope, 4)}, score=score)


# ---------------------------------------------------------------------------
# helpers / registration
# ---------------------------------------------------------------------------

def _round(v: Any, ndigits: int = 4) -> Any:
    if v is None:
        return None
    try:
        return round(float(v), ndigits)
    except (TypeError, ValueError):
        return v


def build_tools() -> list[Tool]:
    return [ObvTool(), VwapTool(), VolumeSpikeTool(), CmfTool(), AdLineTool()]
