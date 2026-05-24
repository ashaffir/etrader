"""Context-aware tools: spread, market hours, higher-TF trend, regime, RS.

Most context tools need state beyond the per-instrument candle
sequence — live rates, exchange schedule heuristics, daily candles
for the same instrument, candles for cross-asset anchors. The cycle
runner builds those into the :class:`ToolContext` once per cycle.

``spread_filter`` and ``market_hours`` are gates: they veto an
otherwise-valid candidate when conditions are unfavorable, saving
both LLM cost and exchange round-trips.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..indicators import momentum_pct, simple_moving_average
from .base import AssetClass, Tool, ToolContext, ToolResult


_PRICE_ASSET_CLASSES = (
    AssetClass.STOCK,
    AssetClass.CRYPTO,
    AssetClass.ETF,
    AssetClass.INDEX,
    AssetClass.COMMODITY,
    AssetClass.FX,
    AssetClass.OTHER,
)


# ---------------------------------------------------------------------------
# Spread filter (gate)
# ---------------------------------------------------------------------------

class SpreadFilterTool(Tool):
    """Veto BUY when the bid-ask spread is wide enough to eat the edge."""

    name = "spread_filter"
    family = "context"
    role = "gate"
    purpose = "Veto BUY when bid-ask spread > 0.5% (configurable)"
    asset_classes = _PRICE_ASSET_CLASSES

    max_spread_pct: float = 0.5

    def evaluate(self, ctx: ToolContext) -> ToolResult:
        rate = ctx.rate
        if rate is None or rate.bid in (None, 0) or rate.ask in (None, 0):
            # No data → don't gate; rate fetch can fail without it being fatal.
            return ToolResult(features={"spread_pct": None})
        mid = (rate.ask + rate.bid) / 2.0
        if mid <= 0:
            return ToolResult(features={"spread_pct": None})
        spread_pct = (rate.ask - rate.bid) / mid * 100.0
        if ctx.candidate_action == "BUY" and spread_pct > self.max_spread_pct:
            return ToolResult(
                features={"spread_pct": round(spread_pct, 4)},
                gate_passed=False,
                gate_reason=f"spread {spread_pct:.2f}% > cap {self.max_spread_pct:.2f}%",
            )
        return ToolResult(features={"spread_pct": round(spread_pct, 4)})


# ---------------------------------------------------------------------------
# Market hours (gate)
# ---------------------------------------------------------------------------

class MarketHoursTool(Tool):
    """Veto when traditional markets are closed (weekend / off-hours).

    Crypto trades 24/7 so it's exempt. For everything else we apply a
    coarse heuristic (US session 13:30-21:00 UTC on weekdays). eToro
    will reject the trade anyway when closed; this gate just avoids
    burning an LLM call.
    """

    name = "market_hours"
    family = "context"
    role = "gate"
    purpose = "Veto trading outside session for non-crypto instruments"
    asset_classes = (
        AssetClass.STOCK, AssetClass.ETF, AssetClass.INDEX,
        AssetClass.COMMODITY, AssetClass.FX, AssetClass.OTHER,
    )

    def evaluate(self, ctx: ToolContext) -> ToolResult:
        now = datetime.now(timezone.utc)
        weekday = now.weekday()  # Mon=0 … Sun=6
        if ctx.asset_class == AssetClass.FX:
            # FX runs ~24x5: closed Sat-Sun only.
            is_open = weekday < 5
        else:
            in_session = (now.hour, now.minute) >= (13, 30) and now.hour < 21
            is_open = weekday < 5 and in_session
        if ctx.candidate_action == "BUY" and not is_open:
            return ToolResult(
                features={"is_open": False, "now_utc": now.isoformat()},
                gate_passed=False,
                gate_reason="market closed",
            )
        return ToolResult(features={"is_open": True})


# ---------------------------------------------------------------------------
# Higher-timeframe trend (feature)
# ---------------------------------------------------------------------------

class HigherTfTrendTool(Tool):
    """Daily-candle trend filter — confirms or warns vs intraday signals."""

    name = "higher_tf_trend"
    family = "context"
    role = "feature"
    purpose = "Daily SMA50 slope: align intraday signal with the bigger trend"
    asset_classes = _PRICE_ASSET_CLASSES

    def evaluate(self, ctx: ToolContext) -> ToolResult:
        daily = list(ctx.higher_tf_candles or [])
        if len(daily) < 60:
            return ToolResult(features={"trend": "unknown"}, score=0.0)
        closes = [c.close for c in daily if c.close > 0]
        sma = simple_moving_average(closes, 50)
        sma_now = sma[-1]
        sma_then = sma[-11] if len(sma) >= 11 else None
        mom = momentum_pct(closes, 10)
        if sma_now is None or sma_then is None or sma_then <= 0:
            return ToolResult(features={"trend": "unknown"}, score=0.0)
        slope_pct = (sma_now - sma_then) / sma_then * 100.0
        trend = "up" if slope_pct > 0.5 else ("down" if slope_pct < -0.5 else "flat")
        score = 0.0
        if trend == "up" and ctx.candidate_action == "BUY":
            score = 0.5
        elif trend == "down" and ctx.candidate_action == "BUY":
            score = -0.5
        elif trend == "down" and ctx.candidate_action == "CLOSE":
            score = 0.4
        return ToolResult(
            features={
                "trend": trend,
                "sma50_slope_pct": round(slope_pct, 3),
                "momentum_10d_pct": round(mom, 3) if mom is not None else None,
            },
            score=score,
        )


# ---------------------------------------------------------------------------
# Cross-asset regime (feature)
# ---------------------------------------------------------------------------

class CrossAssetRegimeTool(Tool):
    """Surface the cycle's risk-on / risk-off snapshot."""

    name = "cross_asset_regime"
    family = "context"
    role = "feature"
    purpose = "Risk-on/off snapshot from SPX500 + BTC anchors"
    asset_classes = _PRICE_ASSET_CLASSES

    def evaluate(self, ctx: ToolContext) -> ToolResult:
        regime = ctx.cross_asset_regime
        if not regime:
            return ToolResult(features={"risk_on": None}, score=0.0)
        risk_on = bool(regime.get("risk_on"))
        score = 0.0
        if ctx.candidate_action == "BUY" and not risk_on:
            score = -0.3
        elif ctx.candidate_action == "BUY" and risk_on:
            score = 0.3
        elif ctx.candidate_action == "CLOSE" and not risk_on:
            score = 0.3
        return ToolResult(features=dict(regime), score=score)


# ---------------------------------------------------------------------------
# Relative strength (feature)
# ---------------------------------------------------------------------------

class RelativeStrengthTool(Tool):
    """Outperformance vs SPX over the last 20 bars (or BTC for crypto)."""

    name = "relative_strength"
    family = "context"
    role = "feature"
    purpose = "Recent outperformance vs SPX (or BTC for crypto)"
    asset_classes = _PRICE_ASSET_CLASSES

    def evaluate(self, ctx: ToolContext) -> ToolResult:
        own_mom = momentum_pct(ctx.closes(), 20)
        anchor_key = "btc_momentum_pct" if ctx.asset_class == AssetClass.CRYPTO else "spx_momentum_pct"
        anchor = (ctx.cross_asset_regime or {}).get(anchor_key)
        if own_mom is None or anchor is None:
            return ToolResult(features={"rel_strength_pct": None}, score=0.0)
        rel = float(own_mom) - float(anchor)
        score = 0.0
        if ctx.candidate_action == "BUY":
            score = max(-0.4, min(0.4, rel / 10.0))
        else:
            score = max(-0.4, min(0.4, -rel / 10.0))
        return ToolResult(
            features={
                "own_momentum_pct": round(own_mom, 3),
                "anchor_momentum_pct": round(float(anchor), 3),
                "rel_strength_pct": round(rel, 3),
            },
            score=score,
        )


def build_tools() -> list[Tool]:
    return [
        SpreadFilterTool(),
        MarketHoursTool(),
        HigherTfTrendTool(),
        CrossAssetRegimeTool(),
        RelativeStrengthTool(),
    ]
