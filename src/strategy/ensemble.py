"""Weighted price-tool ensemble for entry / exit candidacy.

The legacy strategy required SMA cross AND RSI<overbought AND momentum>0
all together to produce a Candidate. That throttled the bot to a single
narrow setup and starved the rest of the tool catalog (MACD, Bollinger,
Donchian, EMA cross, …) of any influence on candidacy itself — they
could only ever enrich or veto candidates that already passed those
three rules.

This module replaces that strict AND with a weighted ensemble. Every
price tool emits a signed component score in [-1, +1]; we aggregate
them, normalize by the total weight, and produce a single
``raw_score`` in [-1, +1]. The signal layer turns positive raw_score
into BUY strength and negative raw_score into CLOSE strength.

Conventions for component scores:
    +1.0   strong bullish trigger fired in the recent window
    -1.0   strong bearish trigger fired in the recent window
    +/-0.3 directional state without a fresh trigger
       0   not enough data, or signal is genuinely flat

Each component returns a ``ComponentScore`` which the caller can
surface to the LLM and the operator so the bot can always say
"these N tools voted bullish, these M voted bearish" rather than
hand-waving.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..config import StrategyConfig
from .indicators.price import (
    bollinger_bands,
    donchian_channel,
    exponential_moving_average,
    macd,
    momentum_pct,
    relative_strength_index,
    simple_moving_average,
)


_CROSS_LOOKBACK = 5      # bars to look back for SMA / EMA crosses
_MACD_LOOKBACK = 3       # bars to look back for MACD histogram zero-crosses


@dataclass(frozen=True)
class ComponentScore:
    """Per-tool signed contribution to the entry ensemble."""

    name: str            # "sma_cross", "ema_cross", "rsi", "macd", "bollinger", "donchian", "momentum"
    score: float         # signed in [-1, +1]; positive = bullish
    detail: str          # short human-readable explanation


@dataclass(frozen=True)
class EnsembleResult:
    """Weighted aggregate of all component scores."""

    raw_score: float                              # in [-1, +1]
    components: tuple[ComponentScore, ...]        # one entry per tool, in stable order
    total_weight: float

    @property
    def buy_strength(self) -> float:
        return max(0.0, self.raw_score)

    @property
    def sell_strength(self) -> float:
        return max(0.0, -self.raw_score)

    def top_contributors(self, *, k: int = 3) -> tuple[ComponentScore, ...]:
        """Return the K components with largest absolute contribution."""
        ordered = sorted(self.components, key=lambda c: abs(c.score), reverse=True)
        return tuple(ordered[:k])


# ---------------------------------------------------------------------------
# Component scorers
# ---------------------------------------------------------------------------

def _score_sma_cross(closes: Sequence[float], cfg: StrategyConfig) -> ComponentScore:
    short = simple_moving_average(closes, cfg.sma_short_period)
    long = simple_moving_average(closes, cfg.sma_long_period)
    if not short or not long or short[-1] is None or long[-1] is None:
        return ComponentScore("sma_cross", 0.0, "insufficient data")
    s_now, l_now = short[-1], long[-1]
    cross_up = _crossed_recently(short, long, _CROSS_LOOKBACK, direction="up")
    cross_down = _crossed_recently(short, long, _CROSS_LOOKBACK, direction="down")
    if cross_up:
        return ComponentScore("sma_cross", 1.0,
                              f"SMA bull cross within {_CROSS_LOOKBACK} bars")
    if cross_down:
        return ComponentScore("sma_cross", -1.0,
                              f"SMA bear cross within {_CROSS_LOOKBACK} bars")
    if l_now <= 0:
        return ComponentScore("sma_cross", 0.0, "no SMA reading")
    spread_pct = (s_now - l_now) / l_now * 100.0
    direction = 0.3 if spread_pct > 0.05 else (-0.3 if spread_pct < -0.05 else 0.0)
    return ComponentScore(
        "sma_cross", direction,
        f"SMA spread {spread_pct:+.2f}% (no recent cross)",
    )


def _score_ema_cross(closes: Sequence[float], cfg: StrategyConfig) -> ComponentScore:
    fast = exponential_moving_average(closes, cfg.ema_fast_period)
    slow = exponential_moving_average(closes, cfg.ema_slow_period)
    if not fast or not slow or fast[-1] is None or slow[-1] is None:
        return ComponentScore("ema_cross", 0.0, "insufficient data")
    cross_up = _crossed_recently(fast, slow, _CROSS_LOOKBACK, direction="up")
    cross_down = _crossed_recently(fast, slow, _CROSS_LOOKBACK, direction="down")
    if cross_up:
        return ComponentScore("ema_cross", 1.0,
                              f"EMA bull cross within {_CROSS_LOOKBACK} bars")
    if cross_down:
        return ComponentScore("ema_cross", -1.0,
                              f"EMA bear cross within {_CROSS_LOOKBACK} bars")
    f_now, s_now = fast[-1], slow[-1]
    if s_now <= 0:
        return ComponentScore("ema_cross", 0.0, "no EMA reading")
    spread_pct = (f_now - s_now) / s_now * 100.0
    direction = 0.3 if spread_pct > 0.05 else (-0.3 if spread_pct < -0.05 else 0.0)
    return ComponentScore(
        "ema_cross", direction,
        f"EMA spread {spread_pct:+.2f}% (no recent cross)",
    )


def _score_rsi(closes: Sequence[float], cfg: StrategyConfig) -> ComponentScore:
    series = relative_strength_index(closes, cfg.rsi_period)
    rsi = series[-1] if series else None
    if rsi is None:
        return ComponentScore("rsi", 0.0, "insufficient data")
    if rsi <= cfg.rsi_oversold:
        return ComponentScore("rsi", 1.0, f"RSI={rsi:.1f} ≤ oversold={cfg.rsi_oversold:.0f}")
    if rsi >= cfg.rsi_overbought:
        return ComponentScore("rsi", -1.0, f"RSI={rsi:.1f} ≥ overbought={cfg.rsi_overbought:.0f}")
    span = cfg.rsi_overbought - cfg.rsi_oversold
    if span <= 0:
        return ComponentScore("rsi", 0.0, "invalid RSI band")
    # Linear from +0.5 at oversold edge to -0.5 at overbought edge
    score = (cfg.rsi_oversold + cfg.rsi_overbought - 2 * rsi) / span
    score = max(-0.5, min(0.5, score))
    return ComponentScore("rsi", round(score, 4), f"RSI={rsi:.1f} (mid-range)")


def _score_macd(closes: Sequence[float], cfg: StrategyConfig) -> ComponentScore:
    macd_line, sig_line, hist = macd(
        closes,
        fast=cfg.macd_fast,
        slow=cfg.macd_slow,
        signal=cfg.macd_signal,
    )
    if not hist or hist[-1] is None:
        return ComponentScore("macd", 0.0, "insufficient data")
    h_now = hist[-1]
    cross_up = _hist_crossed(hist, _MACD_LOOKBACK, direction="up")
    cross_down = _hist_crossed(hist, _MACD_LOOKBACK, direction="down")
    if cross_up:
        return ComponentScore("macd", 1.0,
                              f"MACD bull cross within {_MACD_LOOKBACK} bars")
    if cross_down:
        return ComponentScore("macd", -1.0,
                              f"MACD bear cross within {_MACD_LOOKBACK} bars")
    direction = 0.4 if h_now > 0 else (-0.4 if h_now < 0 else 0.0)
    return ComponentScore("macd", direction, f"MACD hist={h_now:+.4f} (no recent cross)")


def _score_bollinger(closes: Sequence[float], cfg: StrategyConfig) -> ComponentScore:
    lower, middle, upper = bollinger_bands(
        closes, period=cfg.bollinger_period, stddev=cfg.bollinger_stddev,
    )
    if not lower or lower[-1] is None or upper[-1] is None or middle[-1] is None:
        return ComponentScore("bollinger", 0.0, "insufficient data")
    last_close = float(closes[-1])
    lo, mid, up = float(lower[-1]), float(middle[-1]), float(upper[-1])
    if up <= lo:
        return ComponentScore("bollinger", 0.0, "degenerate band")
    if last_close >= up:
        return ComponentScore("bollinger", -1.0,
                              f"close={last_close:.2f} above upper band {up:.2f}")
    if last_close <= lo:
        return ComponentScore("bollinger", 1.0,
                              f"close={last_close:.2f} below lower band {lo:.2f}")
    # Position within band: above midline → mildly bullish, below → mildly bearish
    width = up - lo
    norm = (last_close - mid) / (width / 2)  # in [-1, +1] within the band
    score = max(-0.4, min(0.4, norm * 0.4))
    return ComponentScore(
        "bollinger", round(score, 4),
        f"close inside band ({norm:+.2f} from mid)",
    )


def _score_donchian(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    cfg: StrategyConfig,
) -> ComponentScore:
    lower, upper = donchian_channel(highs, lows, period=cfg.donchian_period)
    # We need the PRIOR bar's channel (excluding the current bar) so a
    # fresh new high actually counts as a breakout, not just touches the
    # channel that was just extended.
    if len(upper) < 2 or upper[-2] is None or lower[-2] is None:
        return ComponentScore("donchian", 0.0, "insufficient data")
    last_close = float(closes[-1])
    prev_up = float(upper[-2])
    prev_lo = float(lower[-2])
    if last_close > prev_up:
        return ComponentScore("donchian", 1.0,
                              f"breakout above {prev_up:.2f}")
    if last_close < prev_lo:
        return ComponentScore("donchian", -1.0,
                              f"breakdown below {prev_lo:.2f}")
    return ComponentScore("donchian", 0.0, "inside channel")


def _score_momentum(closes: Sequence[float], cfg: StrategyConfig) -> ComponentScore:
    mom = momentum_pct(closes, cfg.momentum_lookback)
    if mom is None:
        return ComponentScore("momentum", 0.0, "insufficient data")
    score = max(-1.0, min(1.0, mom / 8.0))   # ~+/-8% saturates
    return ComponentScore("momentum", round(score, 4), f"mom={mom:+.2f}%")


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def evaluate_ensemble(
    *,
    closes: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    cfg: StrategyConfig,
) -> EnsembleResult:
    """Run every component and aggregate into a signed raw score."""
    components = [
        _score_sma_cross(closes, cfg),
        _score_ema_cross(closes, cfg),
        _score_rsi(closes, cfg),
        _score_macd(closes, cfg),
        _score_bollinger(closes, cfg),
        _score_donchian(highs, lows, closes, cfg),
        _score_momentum(closes, cfg),
    ]
    weight_lookup = {
        "sma_cross": cfg.weight_sma_cross,
        "ema_cross": cfg.weight_ema_cross,
        "rsi": cfg.weight_rsi,
        "macd": cfg.weight_macd,
        "bollinger": cfg.weight_bollinger,
        "donchian": cfg.weight_donchian,
        "momentum": cfg.weight_momentum,
    }
    weighted_sum = 0.0
    total_weight = 0.0
    for c in components:
        w = float(weight_lookup.get(c.name, 0.0))
        if w == 0.0:
            continue
        weighted_sum += w * c.score
        total_weight += abs(w)
    raw = weighted_sum / total_weight if total_weight > 0 else 0.0
    return EnsembleResult(
        raw_score=round(raw, 4),
        components=tuple(components),
        total_weight=total_weight,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _crossed_recently(
    short_series: Sequence[float | None],
    long_series: Sequence[float | None],
    lookback: int,
    *,
    direction: str,
) -> bool:
    """Return True iff series crossed `direction` within last `lookback` bars."""
    if not short_series or not long_series:
        return False
    s_now = short_series[-1]
    l_now = long_series[-1]
    if s_now is None or l_now is None:
        return False
    if direction == "up" and s_now <= l_now:
        return False
    if direction == "down" and s_now >= l_now:
        return False
    upper = min(len(short_series), len(long_series))
    for offset in range(2, min(lookback + 2, upper + 1)):
        s_prev = short_series[-offset]
        l_prev = long_series[-offset]
        if s_prev is None or l_prev is None:
            continue
        if direction == "up" and s_prev <= l_prev:
            return True
        if direction == "down" and s_prev >= l_prev:
            return True
    return False


def _hist_crossed(
    histogram: Sequence[float | None],
    lookback: int,
    *,
    direction: str,
) -> bool:
    """Return True iff MACD histogram crossed zero in the last `lookback` bars."""
    if not histogram or histogram[-1] is None:
        return False
    h_now = histogram[-1]
    if direction == "up" and h_now <= 0:
        return False
    if direction == "down" and h_now >= 0:
        return False
    upper = len(histogram)
    for offset in range(2, min(lookback + 2, upper + 1)):
        h_prev = histogram[-offset]
        if h_prev is None:
            continue
        if direction == "up" and h_prev <= 0:
            return True
        if direction == "down" and h_prev >= 0:
            return True
    return False
