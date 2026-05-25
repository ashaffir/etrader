"""Universe activity filter — drop symbols that aren't actually tradeable.

The news pipeline produces *candidates* — tickers with a reason to be
watched. Not every candidate is worth a slot in the tracked universe:

* If a symbol's price is too **flat** (low ATR), the bot's SL/TP
  thresholds will never trigger and we'll just incur idle data fetches.
* If a symbol's **spread** is too wide (ask−bid as % of mid), the
  cost-to-trade swallows the expected edge — every BUY would lock in
  a loss before the indicator-driven exit fires.

This module implements those two gates as pure functions over a
:class:`~src.etoro.market_data.Candle` sequence and a
:class:`~src.etoro.market_data.LiveRate`. The result is a structured
:class:`ActivityDecision` carrying both the pass/fail verdict and the
underlying numbers — surfaced through the `/universe` and `/news`
Telegram commands so operators can audit rejections at a glance.

The filter has zero I/O and zero state; instantiate once, call
:meth:`ActivityFilter.evaluate` per (symbol, candles, rate).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..config import UniverseConfig
from ..etoro.market_data import Candle, LiveRate
from .indicators import average_true_range


@dataclass(frozen=True)
class ActivityDecision:
    """Verdict + measured values for one candidate.

    Attributes
    ----------
    passed:
        ``True`` if the candidate cleared every gate, ``False`` otherwise.
    reason:
        Short, human-readable explanation. Always populated — pass
        reasons describe *why* the symbol qualified (e.g. ``"ok atr=0.8%
        spread=0.12%"``) so the universe summary can carry the metrics
        without a second lookup.
    atr_pct:
        Wilder ATR over ``atr_period`` divided by last close, ×100.
        ``None`` when not enough candles were available.
    spread_pct:
        ``(ask - bid) / mid`` × 100. ``None`` when the rate had a
        missing leg (eToro occasionally returns asymmetric quotes for
        thinly-traded crypto pairs).
    """

    passed: bool
    reason: str
    atr_pct: float | None
    spread_pct: float | None

    def short_summary(self) -> str:
        """Compact "atr=… spread=…" line for log/UI."""
        atr = f"{self.atr_pct:.2f}%" if self.atr_pct is not None else "n/a"
        spr = f"{self.spread_pct:.3f}%" if self.spread_pct is not None else "n/a"
        return f"atr={atr} spread={spr}"


class ActivityFilter:
    """Stateless ATR%/spread% gate over a single candidate.

    The thresholds come from :class:`~src.config.UniverseConfig`:

    * ``min_atr_pct``: minimum ATR% — below this we judge the symbol
      too flat to clear typical SL/TP and reject.
    * ``max_spread_pct``: maximum spread% — above this the round-trip
      cost is unacceptable.
    * ``atr_period``: Wilder ATR window (default 14 bars).
    * ``activity_min_candles``: minimum candles to evaluate; we'd
      rather reject (``"insufficient_candles"``) than guess on a tiny
      window.

    The filter is intentionally **soft** on missing rate data: if
    ``spread_pct`` is unknown, the spread gate is skipped (but ATR is
    still required). This matches the reality that eToro sometimes
    fails to populate ask/bid for a fast-moving instrument while the
    candle history is intact.
    """

    def __init__(self, cfg: UniverseConfig) -> None:
        self._min_atr_pct = float(cfg.min_atr_pct)
        self._max_spread_pct = float(cfg.max_spread_pct)
        self._atr_period = int(cfg.atr_period)
        self._min_candles = max(int(cfg.activity_min_candles), self._atr_period + 1)

    # ------------------------------------------------------------------

    def evaluate(
        self,
        *,
        candles: Sequence[Candle],
        rate: LiveRate | None,
    ) -> ActivityDecision:
        """Run both gates over the supplied data."""
        atr_pct = self._atr_pct(candles)
        spread_pct = self._spread_pct(rate)

        if atr_pct is None:
            return ActivityDecision(
                passed=False,
                reason=f"insufficient candles (<{self._min_candles})",
                atr_pct=None,
                spread_pct=spread_pct,
            )
        if atr_pct < self._min_atr_pct:
            return ActivityDecision(
                passed=False,
                reason=(
                    f"flat: atr={atr_pct:.2f}% < min={self._min_atr_pct:.2f}%"
                ),
                atr_pct=atr_pct,
                spread_pct=spread_pct,
            )
        if spread_pct is not None and spread_pct > self._max_spread_pct:
            return ActivityDecision(
                passed=False,
                reason=(
                    f"wide spread: {spread_pct:.3f}% > "
                    f"max={self._max_spread_pct:.3f}%"
                ),
                atr_pct=atr_pct,
                spread_pct=spread_pct,
            )
        return ActivityDecision(
            passed=True,
            reason=f"ok atr={atr_pct:.2f}% spread={spread_pct:.3f}%"
            if spread_pct is not None
            else f"ok atr={atr_pct:.2f}% spread=n/a",
            atr_pct=atr_pct,
            spread_pct=spread_pct,
        )

    # ------------------------------------------------------------------

    def _atr_pct(self, candles: Sequence[Candle]) -> float | None:
        if len(candles) < self._min_candles:
            return None
        # Filter out any rows where high/low/close are zero/None — they
        # blow up the ATR calc and indicate eToro returned an inactive
        # bar (weekend, halt, pre-listing).
        clean = [c for c in candles if c.high > 0 and c.low > 0 and c.close > 0]
        if len(clean) < self._min_candles:
            return None
        highs = [c.high for c in clean]
        lows = [c.low for c in clean]
        closes = [c.close for c in clean]
        atr = average_true_range(highs, lows, closes, period=self._atr_period)
        if atr is None or closes[-1] <= 0:
            return None
        return atr / closes[-1] * 100.0

    @staticmethod
    def _spread_pct(rate: LiveRate | None) -> float | None:
        if rate is None:
            return None
        ask, bid = rate.ask, rate.bid
        if ask is None or bid is None or ask <= 0 or bid <= 0:
            return None
        if ask <= bid:
            # Inverted spread → treat as missing; eToro occasionally
            # returns stale legs.
            return None
        mid = (ask + bid) / 2.0
        if mid <= 0:
            return None
        return (ask - bid) / mid * 100.0
