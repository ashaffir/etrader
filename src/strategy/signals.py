"""Deterministic technical signals built from candle history.

The signal layer turns raw OHLCV into a sorted shortlist of
``Candidate``s the LLM can reason about. It is the single entry point
into trade-decision land — only symbols that earn a deterministic
candidacy here can ever reach the LLM or the executor.

Entry / exit are decided by a **weighted price-tool ensemble** (see
:mod:`src.strategy.ensemble`). Each registered price tool — SMA cross,
EMA cross, RSI, MACD, Bollinger, Donchian, momentum — emits a signed
score. We aggregate them into a single ``raw_score`` in [-1, +1]:

- ``raw_score >= cfg.min_signal_strength``  → **BUY** Candidate
  (only if the instrument is *not* already bot-owned).
- ``raw_score <= -cfg.min_exit_strength``   → **CLOSE** Candidate
  (only for instruments the bot already owns).

The full per-component breakdown is attached to the Candidate so the
LLM and the operator can see exactly which tools voted bullish vs
bearish — no more "which tool decided this?" guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from ..config import StrategyConfig
from ..etoro.market_data import Candle
from .ensemble import ComponentScore, EnsembleResult, evaluate_ensemble


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    instrument_id: int
    symbol: str
    action: str                 # "BUY" | "CLOSE"
    strength: float             # 0..1 (signed magnitude of the ensemble)
    reason: str
    last_close: float
    rsi: float | None
    sma_short: float | None
    sma_long: float | None
    momentum_pct: float | None
    raw_score: float = 0.0      # signed ensemble raw_score, in [-1, +1]
    components: tuple[ComponentScore, ...] = field(default_factory=tuple)

    def components_dict(self) -> dict[str, dict[str, float | str]]:
        """Per-component breakdown the LLM/operator can quote verbatim."""
        return {
            c.name: {"score": c.score, "detail": c.detail}
            for c in self.components
        }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_candidates(
    *,
    cfg: StrategyConfig,
    candles_by_instrument: Mapping[int, Sequence[Candle]],
    symbol_for_id: Mapping[int, str],
    bot_owned_instrument_ids: Iterable[int],
) -> list[Candidate]:
    owned = set(bot_owned_instrument_ids)
    out: list[Candidate] = []
    min_bars = _min_bars_required(cfg)
    for inst_id, candles in candles_by_instrument.items():
        if not candles:
            continue
        if len(candles) < min_bars:
            continue
        candidate = _evaluate(
            cfg=cfg,
            instrument_id=inst_id,
            symbol=symbol_for_id.get(inst_id, str(inst_id)),
            candles=candles,
            owns_position=inst_id in owned,
        )
        if candidate is not None:
            out.append(candidate)
    out.sort(key=lambda c: c.strength, reverse=True)
    return out


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _min_bars_required(cfg: StrategyConfig) -> int:
    """Cheapest computable safe minimum across every component."""
    return max(
        cfg.sma_long_period,
        cfg.ema_slow_period,
        cfg.rsi_period + 1,
        cfg.macd_slow + cfg.macd_signal,
        cfg.bollinger_period,
        cfg.donchian_period + 1,         # need prior-bar channel
        cfg.momentum_lookback + 1,
    )


def _evaluate(
    *,
    cfg: StrategyConfig,
    instrument_id: int,
    symbol: str,
    candles: Sequence[Candle],
    owns_position: bool,
) -> Candidate | None:
    closes = [c.close for c in candles if c.close > 0]
    highs = [c.high if c.high > 0 else c.close for c in candles]
    lows = [c.low if c.low > 0 else c.close for c in candles]
    if not closes:
        return None

    ensemble = evaluate_ensemble(closes=closes, highs=highs, lows=lows, cfg=cfg)

    last_close = float(closes[-1])
    rsi_now, sma_short_now, sma_long_now, mom_now = _legacy_indicator_snapshot(closes, cfg)

    if owns_position:
        if ensemble.sell_strength >= cfg.min_exit_strength:
            return Candidate(
                instrument_id=instrument_id,
                symbol=symbol,
                action="CLOSE",
                strength=round(ensemble.sell_strength, 4),
                reason=_format_reason(ensemble, sign="negative"),
                last_close=last_close,
                rsi=rsi_now,
                sma_short=sma_short_now,
                sma_long=sma_long_now,
                momentum_pct=mom_now,
                raw_score=ensemble.raw_score,
                components=ensemble.components,
            )
        return None

    # Unowned instrument → BUY candidacy only
    if ensemble.buy_strength >= cfg.min_signal_strength:
        return Candidate(
            instrument_id=instrument_id,
            symbol=symbol,
            action="BUY",
            strength=round(ensemble.buy_strength, 4),
            reason=_format_reason(ensemble, sign="positive"),
            last_close=last_close,
            rsi=rsi_now,
            sma_short=sma_short_now,
            sma_long=sma_long_now,
            momentum_pct=mom_now,
            raw_score=ensemble.raw_score,
            components=ensemble.components,
        )
    return None


def _legacy_indicator_snapshot(
    closes: Sequence[float],
    cfg: StrategyConfig,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Compute the four "headline" indicators for the Candidate envelope.

    Kept around because :class:`~src.strategy.decisions.DecisionEngine`
    and the operator-facing /signals output still display rsi, SMA
    short/long, and momentum verbatim.
    """
    from .indicators.price import (
        momentum_pct,
        relative_strength_index,
        simple_moving_average,
    )

    sma_short_series = simple_moving_average(closes, cfg.sma_short_period)
    sma_long_series = simple_moving_average(closes, cfg.sma_long_period)
    rsi_series = relative_strength_index(closes, cfg.rsi_period)
    return (
        rsi_series[-1] if rsi_series else None,
        sma_short_series[-1] if sma_short_series else None,
        sma_long_series[-1] if sma_long_series else None,
        momentum_pct(closes, cfg.momentum_lookback),
    )


def _format_reason(ensemble: EnsembleResult, *, sign: str) -> str:
    """Render the top contributors for the operator-facing reason field."""
    contributors = ensemble.top_contributors(k=3)
    if sign == "positive":
        relevant = [c for c in contributors if c.score > 0]
    else:
        relevant = [c for c in contributors if c.score < 0]
    if not relevant:
        relevant = list(contributors)
    bits = [f"{c.name}({c.score:+.2f}): {c.detail}" for c in relevant]
    return " | ".join(bits)
