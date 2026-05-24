"""Price-only indicators: SMA, EMA, RSI, momentum, ATR, MACD, Bollinger, Donchian.

All functions accept a sequence of values, oldest first, and return
either a single float (latest value) or a list of floats aligned with
the input (``None`` where insufficient data). Volume-based variants
live in :mod:`.volume`.
"""

from __future__ import annotations

from typing import Sequence


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------

def simple_moving_average(values: Sequence[float], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("period must be positive")
    out: list[float | None] = []
    rolling = 0.0
    for i, v in enumerate(values):
        rolling += v
        if i >= period:
            rolling -= values[i - period]
        if i + 1 >= period:
            out.append(rolling / period)
        else:
            out.append(None)
    return out


def exponential_moving_average(values: Sequence[float], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("period must be positive")
    out: list[float | None] = []
    if not values:
        return out
    multiplier = 2.0 / (period + 1)
    ema: float | None = None
    for i, v in enumerate(values):
        if i + 1 < period:
            out.append(None)
            continue
        if ema is None:
            seed = sum(values[i + 1 - period : i + 1]) / period
            ema = seed
            out.append(ema)
        else:
            ema = (v - ema) * multiplier + ema
            out.append(ema)
    return out


# ---------------------------------------------------------------------------
# Oscillators
# ---------------------------------------------------------------------------

def relative_strength_index(values: Sequence[float], period: int = 14) -> list[float | None]:
    """Wilder's RSI."""
    if period <= 0:
        raise ValueError("period must be positive")
    out: list[float | None] = [None] * len(values)
    if len(values) <= period:
        return out
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        change = values[i] - values[i - 1]
        if change >= 0:
            gains += change
        else:
            losses += -change
    avg_gain = gains / period
    avg_loss = losses / period
    out[period] = _rsi_from(avg_gain, avg_loss)
    for i in range(period + 1, len(values)):
        change = values[i] - values[i - 1]
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        out[i] = _rsi_from(avg_gain, avg_loss)
    return out


def _rsi_from(avg_gain: float, avg_loss: float) -> float:
    # No movement at all → neutral, not "overbought". Wilder's classic
    # formula returns 100 when avg_loss == 0 because that signals
    # "all gains, no losses" — but that's only true when avg_gain > 0.
    if avg_loss == 0:
        return 50.0 if avg_gain == 0 else 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def momentum_pct(values: Sequence[float], lookback: int) -> float | None:
    """Percent change between latest and price ``lookback`` candles ago."""
    if lookback <= 0 or len(values) <= lookback:
        return None
    base = values[-1 - lookback]
    if base == 0:
        return None
    return (values[-1] - base) / base * 100.0


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def macd(
    values: Sequence[float],
    *,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """Return ``(macd_line, signal_line, histogram)`` aligned with input.

    Lines are ``None`` where insufficient lookback is available.
    """
    fast_ema = exponential_moving_average(values, fast)
    slow_ema = exponential_moving_average(values, slow)
    macd_line: list[float | None] = []
    for f, s in zip(fast_ema, slow_ema):
        if f is None or s is None:
            macd_line.append(None)
        else:
            macd_line.append(f - s)
    valid = [v for v in macd_line if v is not None]
    if not valid:
        return macd_line, [None] * len(values), [None] * len(values)
    sig_compact = exponential_moving_average(valid, signal)
    signal_line: list[float | None] = []
    j = 0
    for v in macd_line:
        if v is None:
            signal_line.append(None)
        else:
            signal_line.append(sig_compact[j])
            j += 1
    histogram: list[float | None] = []
    for m, s in zip(macd_line, signal_line):
        if m is None or s is None:
            histogram.append(None)
        else:
            histogram.append(m - s)
    return macd_line, signal_line, histogram


# ---------------------------------------------------------------------------
# Volatility envelopes
# ---------------------------------------------------------------------------

def bollinger_bands(
    values: Sequence[float],
    *,
    period: int = 20,
    stddev: float = 2.0,
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """Return ``(lower, middle, upper)`` Bollinger Bands aligned with input."""
    if period <= 0:
        raise ValueError("period must be positive")
    middle = simple_moving_average(values, period)
    lower: list[float | None] = []
    upper: list[float | None] = []
    for i, m in enumerate(middle):
        if m is None or i + 1 < period:
            lower.append(None)
            upper.append(None)
            continue
        window = values[i + 1 - period : i + 1]
        var = sum((x - m) ** 2 for x in window) / period
        sd = var ** 0.5
        lower.append(m - stddev * sd)
        upper.append(m + stddev * sd)
    return lower, middle, upper


def donchian_channel(
    highs: Sequence[float],
    lows: Sequence[float],
    *,
    period: int = 20,
) -> tuple[list[float | None], list[float | None]]:
    """Return ``(lower, upper)`` rolling min-low/max-high over ``period``."""
    if period <= 0:
        raise ValueError("period must be positive")
    n = min(len(highs), len(lows))
    upper: list[float | None] = [None] * n
    lower: list[float | None] = [None] * n
    for i in range(n):
        if i + 1 < period:
            continue
        window_h = highs[i + 1 - period : i + 1]
        window_l = lows[i + 1 - period : i + 1]
        upper[i] = max(window_h)
        lower[i] = min(window_l)
    return lower, upper


def average_true_range(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> float | None:
    """Wilder's ATR — used as a volatility proxy when sizing SL/TP."""
    n = min(len(highs), len(lows), len(closes))
    if n < period + 1:
        return None
    trs: list[float] = []
    for i in range(1, n):
        prev_close = closes[i - 1]
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - prev_close),
            abs(lows[i] - prev_close),
        )
        trs.append(tr)
    if len(trs) < period:
        return None
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr
