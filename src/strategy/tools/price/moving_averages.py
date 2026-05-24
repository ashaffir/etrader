"""SMA / EMA cross tools — both feature-only.

Both detect a recent (within 5 bars) cross of a fast moving average
above or below a slow moving average and emit a directional score
matched to the candidate's action.
"""

from __future__ import annotations

from typing import Any, Sequence

from ...indicators import (
    exponential_moving_average,
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


def _crossed_recently(
    short: Sequence[float | None],
    long: Sequence[float | None],
    lookback: int = 5,
    *,
    up: bool,
) -> bool:
    """Return True iff ``short`` crossed ``up``/``down`` against ``long`` recently."""
    if not short or not long:
        return False
    s_now, l_now = short[-1], long[-1]
    if s_now is None or l_now is None:
        return False
    if up and s_now <= l_now:
        return False
    if not up and s_now >= l_now:
        return False
    upper = min(len(short), len(long))
    for offset in range(2, min(lookback + 2, upper + 1)):
        s_prev = short[-offset]
        l_prev = long[-offset]
        if s_prev is None or l_prev is None:
            continue
        if up and s_prev <= l_prev:
            return True
        if not up and s_prev >= l_prev:
            return True
    return False


def _round(v: Any, ndigits: int = 4) -> Any:
    if v is None:
        return None
    try:
        return round(float(v), ndigits)
    except (TypeError, ValueError):
        return v


class SmaCrossTool(Tool):
    name = "sma_cross"
    family = "price"
    role = "feature"
    purpose = "SMA(short) vs SMA(long) cross direction within last 5 bars"
    asset_classes = _PRICE_ASSET_CLASSES

    def evaluate(self, ctx: ToolContext) -> ToolResult:
        closes = ctx.closes()
        short = simple_moving_average(closes, ctx.strategy.sma_short_period)
        long = simple_moving_average(closes, ctx.strategy.sma_long_period)
        crossed_up = _crossed_recently(short, long, up=True)
        crossed_down = _crossed_recently(short, long, up=False)
        last_short = short[-1] if short else None
        last_long = long[-1] if long else None
        gap_pct: float | None = None
        if last_short is not None and last_long not in (None, 0):
            gap_pct = (last_short - last_long) / last_long * 100.0  # type: ignore[operator]
        score = 0.0
        if crossed_up:
            score = 0.7 if ctx.candidate_action == "BUY" else -0.4
        elif crossed_down:
            score = 0.7 if ctx.candidate_action == "CLOSE" else -0.4
        return ToolResult(
            features={
                "crossed_up": crossed_up,
                "crossed_down": crossed_down,
                "gap_pct": _round(gap_pct),
                "sma_short": _round(last_short),
                "sma_long": _round(last_long),
            },
            score=score,
        )


class EmaCrossTool(Tool):
    name = "ema_cross"
    family = "price"
    role = "feature"
    purpose = "EMA(12) vs EMA(26) cross — faster than SMA cross"
    asset_classes = _PRICE_ASSET_CLASSES

    def evaluate(self, ctx: ToolContext) -> ToolResult:
        closes = ctx.closes()
        short = exponential_moving_average(closes, 12)
        long = exponential_moving_average(closes, 26)
        crossed_up = _crossed_recently(short, long, up=True)
        crossed_down = _crossed_recently(short, long, up=False)
        s_now = short[-1] if short else None
        l_now = long[-1] if long else None
        score = 0.0
        if crossed_up:
            score = 0.6 if ctx.candidate_action == "BUY" else -0.3
        elif crossed_down:
            score = 0.6 if ctx.candidate_action == "CLOSE" else -0.3
        return ToolResult(
            features={
                "crossed_up": crossed_up,
                "crossed_down": crossed_down,
                "ema_fast": _round(s_now),
                "ema_slow": _round(l_now),
            },
            score=score,
        )
