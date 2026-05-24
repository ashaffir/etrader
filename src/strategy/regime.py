"""Market regime detection — instrument-level and cross-asset.

The selector uses regime to bias which tools run first. We detect
two things:

- **Instrument regime**: trending vs. ranging, derived from the ratio
  of |close - SMA50| to ATR(14). When price is far from its mean
  measured in ATRs, the move is trending; close to mean = ranging.

- **Cross-asset regime**: a coarse risk-on / risk-off read built
  from anchor instruments (defaults to SPX500 + BTC). Helpful so
  the LLM and selector know whether to treat momentum confirmations
  more or less sceptically.

Both are pure functions over candle sequences — no I/O. The cycle
runner fetches the anchor candles once per cycle and feeds them in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..etoro.market_data import Candle
from .indicators import (
    average_true_range,
    momentum_pct,
    simple_moving_average,
)


# ---------------------------------------------------------------------------
# Instrument regime
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InstrumentRegime:
    label: str           # "trending" | "ranging" | "unknown"
    atr_distance: float | None    # |price - SMA50| / ATR
    sma_slope_pct: float | None   # % change of SMA50 over last 10 bars
    momentum_pct: float | None


def detect_instrument_regime(
    candles: Sequence[Candle],
    *,
    sma_period: int = 50,
    atr_period: int = 14,
    momentum_lookback: int = 10,
    trending_threshold: float = 1.5,
) -> InstrumentRegime:
    """Classify trending vs ranging using ATR-normalized distance from SMA.

    ``trending_threshold`` is in ATR units. 1.5 means: if price is at
    least 1.5 ATRs away from its 50-bar mean, treat as trending. This
    is a deliberately conservative threshold; we'd rather miss an
    early trend than wrongly classify a chop range as a trend.
    """
    closes = [c.close for c in candles if c.close > 0]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    if len(closes) < max(sma_period, atr_period + 1, momentum_lookback + 1):
        return InstrumentRegime("unknown", None, None, None)

    sma_series = simple_moving_average(closes, sma_period)
    sma_now = sma_series[-1]
    atr = average_true_range(highs, lows, closes, atr_period)
    mom = momentum_pct(closes, momentum_lookback)
    if sma_now is None or atr is None or atr <= 0:
        return InstrumentRegime("unknown", None, None, mom)

    atr_distance = abs(closes[-1] - sma_now) / atr
    earlier_sma = sma_series[-1 - momentum_lookback] if len(sma_series) > momentum_lookback else None
    sma_slope_pct = None
    if earlier_sma is not None and earlier_sma > 0:
        sma_slope_pct = (sma_now - earlier_sma) / earlier_sma * 100.0

    if atr_distance >= trending_threshold:
        label = "trending"
    elif atr_distance <= 0.5:
        label = "ranging"
    else:
        # In-between: lean on slope direction so the selector still gets a clear hint.
        label = "trending" if (sma_slope_pct or 0.0) != 0 and abs(sma_slope_pct or 0.0) > 0.5 else "ranging"
    return InstrumentRegime(label, atr_distance, sma_slope_pct, mom)


# ---------------------------------------------------------------------------
# Cross-asset regime
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CrossAssetRegime:
    """Snapshot computed once per cycle, shared by every (instrument, tool)."""

    risk_on: bool
    spx_trend: str        # "up" | "down" | "flat" | "unknown"
    btc_trend: str
    spx_momentum_pct: float | None
    btc_momentum_pct: float | None
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "risk_on": self.risk_on,
            "spx_trend": self.spx_trend,
            "btc_trend": self.btc_trend,
            "spx_momentum_pct": self.spx_momentum_pct,
            "btc_momentum_pct": self.btc_momentum_pct,
            "detail": self.detail,
        }


def detect_cross_asset_regime(
    *,
    spx_candles: Sequence[Candle],
    btc_candles: Sequence[Candle],
    momentum_lookback: int = 20,
    trend_window: int = 50,
) -> CrossAssetRegime:
    """Build a coarse risk-on / risk-off classification from SPX + BTC.

    Risk-on iff at least one of (SPX, BTC) is up-trending and neither
    is down-trending. The exact thresholds are intentionally loose so
    the regime is a *bias*, not a hard gate.
    """
    spx_trend, spx_mom = _trend_for(spx_candles, trend_window, momentum_lookback)
    btc_trend, btc_mom = _trend_for(btc_candles, trend_window, momentum_lookback)
    risk_on = (
        ("up" in (spx_trend, btc_trend))
        and "down" not in (spx_trend, btc_trend)
    )
    detail = f"SPX={spx_trend}({_fmt(spx_mom)}%), BTC={btc_trend}({_fmt(btc_mom)}%)"
    return CrossAssetRegime(
        risk_on=risk_on,
        spx_trend=spx_trend,
        btc_trend=btc_trend,
        spx_momentum_pct=spx_mom,
        btc_momentum_pct=btc_mom,
        detail=detail,
    )


def _trend_for(
    candles: Sequence[Candle],
    trend_window: int,
    momentum_lookback: int,
) -> tuple[str, float | None]:
    closes = [c.close for c in candles if c.close > 0]
    if len(closes) < max(trend_window, momentum_lookback + 1):
        return "unknown", None
    sma = simple_moving_average(closes, trend_window)
    sma_now = sma[-1]
    sma_then = sma[-1 - momentum_lookback] if len(sma) > momentum_lookback else None
    mom = momentum_pct(closes, momentum_lookback)
    if sma_now is None or sma_then is None:
        return "unknown", mom
    delta_pct = (sma_now - sma_then) / sma_then * 100.0 if sma_then > 0 else 0.0
    if delta_pct > 0.5 and (mom or 0) > 0:
        return "up", mom
    if delta_pct < -0.5 and (mom or 0) < 0:
        return "down", mom
    return "flat", mom


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:+.1f}"


def regime_label_from(snapshot: Mapping[str, object] | InstrumentRegime | None) -> str:
    """Coerce a regime dict / dataclass / None into a label the selector understands."""
    if snapshot is None:
        return "trending"
    if isinstance(snapshot, InstrumentRegime):
        return snapshot.label if snapshot.label != "unknown" else "trending"
    label = str(snapshot.get("label") or "trending")
    return label if label in {"trending", "ranging"} else "trending"
