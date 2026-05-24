"""Volume-based indicators: OBV, VWAP, volume_spike, CMF, A/D line.

All take separate close/volume sequences (and high/low where needed)
so the caller can opt out cleanly when volume data isn't available
(eToro returns 0 volume for many FX pairs and indices).
"""

from __future__ import annotations

from typing import Sequence


def has_volume(volumes: Sequence[float], min_nonzero: int = 5) -> bool:
    """Heuristic: do we have enough non-zero volume to compute volume tools?

    eToro frequently returns 0 volume for FX/indices. We require at
    least ``min_nonzero`` non-zero values before any volume tool runs.
    """
    return sum(1 for v in volumes if v and v > 0) >= min_nonzero


def on_balance_volume(closes: Sequence[float], volumes: Sequence[float]) -> list[float | None]:
    """Granville's OBV: cumulative signed volume."""
    n = min(len(closes), len(volumes))
    if n == 0:
        return []
    out: list[float | None] = [0.0]
    for i in range(1, n):
        prev_close = closes[i - 1]
        cur_close = closes[i]
        prev_obv = out[-1] or 0.0
        if cur_close > prev_close:
            out.append(prev_obv + volumes[i])
        elif cur_close < prev_close:
            out.append(prev_obv - volumes[i])
        else:
            out.append(prev_obv)
    return out


def vwap(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Sequence[float],
) -> float | None:
    """Volume-weighted average price over the supplied window.

    Returns ``None`` if we lack volume or the window is empty. Uses
    typical price = (H + L + C) / 3 per candle, weighted by volume.
    """
    n = min(len(highs), len(lows), len(closes), len(volumes))
    if n == 0:
        return None
    weighted_sum = 0.0
    vol_sum = 0.0
    for i in range(n):
        v = volumes[i] or 0.0
        if v <= 0:
            continue
        typical = (highs[i] + lows[i] + closes[i]) / 3.0
        weighted_sum += typical * v
        vol_sum += v
    if vol_sum <= 0:
        return None
    return weighted_sum / vol_sum


def volume_spike_ratio(volumes: Sequence[float], lookback: int = 20) -> float | None:
    """Latest-bar volume divided by the SMA of the prior ``lookback`` bars.

    A value of 2.0 means the last bar had 2x the recent average volume.
    Returns ``None`` if data is insufficient or volume is missing.
    """
    if lookback <= 0 or len(volumes) < lookback + 1:
        return None
    last = volumes[-1] or 0.0
    if last <= 0:
        return None
    window = volumes[-lookback - 1 : -1]
    pos = [v for v in window if v and v > 0]
    if len(pos) < max(3, lookback // 2):
        return None
    avg = sum(pos) / len(pos)
    if avg <= 0:
        return None
    return last / avg


def chaikin_money_flow(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Sequence[float],
    period: int = 20,
) -> float | None:
    """CMF over the trailing ``period`` bars; range -1..+1.

    Positive values indicate accumulation (buying pressure); negative
    values indicate distribution. Returns ``None`` if we don't have
    enough non-zero volume bars.
    """
    n = min(len(highs), len(lows), len(closes), len(volumes))
    if n < period:
        return None
    window_h = highs[-period:]
    window_l = lows[-period:]
    window_c = closes[-period:]
    window_v = volumes[-period:]
    money_flow_volume = 0.0
    total_volume = 0.0
    for h, low, c, v in zip(window_h, window_l, window_c, window_v):
        v = v or 0.0
        if v <= 0 or h == low:
            continue
        mfm = ((c - low) - (h - c)) / (h - low)
        money_flow_volume += mfm * v
        total_volume += v
    if total_volume <= 0:
        return None
    return money_flow_volume / total_volume


def accumulation_distribution_line(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Sequence[float],
) -> list[float | None]:
    """Cumulative A/D line. Useful for trend confirmation: rising A/D
    alongside rising price = healthy uptrend.
    """
    n = min(len(highs), len(lows), len(closes), len(volumes))
    out: list[float | None] = []
    running = 0.0
    for i in range(n):
        h, low, c = highs[i], lows[i], closes[i]
        v = volumes[i] or 0.0
        if v <= 0 or h == low:
            out.append(running)
            continue
        mfm = ((c - low) - (h - c)) / (h - low)
        running += mfm * v
        out.append(running)
    return out
