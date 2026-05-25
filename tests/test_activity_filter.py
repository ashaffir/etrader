"""Tests for the universe activity filter.

The filter is pure: feed it candles + an optional rate, get back an
:class:`ActivityDecision`. We synthesize candle histories with known
ATR characteristics rather than relying on fixtures.
"""

import unittest

from src.config import UniverseConfig
from src.etoro.market_data import Candle, LiveRate
from src.strategy.activity_filter import ActivityDecision, ActivityFilter


def _candles(closes: list[float], *, hl_spread_pct: float = 0.5) -> list[Candle]:
    """Build a candle list around ``closes`` with given (high-low) spread.

    Spread is symmetric around close: high = close * (1 + hl_spread/2),
    low = close * (1 - hl_spread/2). Volume held at 1.0.
    """
    out: list[Candle] = []
    for i, c in enumerate(closes):
        high = c * (1.0 + hl_spread_pct / 200.0)
        low = c * (1.0 - hl_spread_pct / 200.0)
        out.append(
            Candle(
                instrument_id=1,
                from_date=None,
                open=c,
                high=high,
                low=low,
                close=c,
                volume=1.0,
            )
        )
    return out


def _rate(ask: float | None, bid: float | None) -> LiveRate:
    return LiveRate(instrument_id=1, ask=ask, bid=bid, last=None, timestamp=None)


def _cfg(**overrides) -> UniverseConfig:
    return UniverseConfig(**overrides)


class ActivityFilterTests(unittest.TestCase):
    def test_passes_when_atr_above_min_and_spread_below_max(self) -> None:
        # Volatile candles (1% high-low) → ATR% ≈ 1%; tight spread.
        candles = _candles([100.0 + 0.5 * i for i in range(30)], hl_spread_pct=1.0)
        rate = _rate(ask=100.10, bid=100.00)
        decision = ActivityFilter(_cfg(min_atr_pct=0.3, max_spread_pct=1.0)).evaluate(
            candles=candles, rate=rate,
        )
        self.assertTrue(decision.passed, decision.reason)
        self.assertIsNotNone(decision.atr_pct)
        self.assertGreater(decision.atr_pct or 0.0, 0.3)
        self.assertIn("ok atr=", decision.reason)

    def test_rejects_low_atr(self) -> None:
        # Completely flat candles (close=high=low) → ATR ≈ 0.
        flat = [
            Candle(instrument_id=1, from_date=None, open=100.0, high=100.0,
                   low=100.0, close=100.0, volume=1.0)
            for _ in range(30)
        ]
        decision = ActivityFilter(_cfg(min_atr_pct=0.3)).evaluate(
            candles=flat, rate=_rate(100.05, 100.00),
        )
        self.assertFalse(decision.passed)
        self.assertIn("flat", decision.reason)

    def test_rejects_wide_spread(self) -> None:
        candles = _candles([100.0 + 0.5 * i for i in range(30)], hl_spread_pct=1.0)
        # 2% spread, way above 1% cap.
        rate = _rate(ask=101.0, bid=99.0)
        decision = ActivityFilter(_cfg(min_atr_pct=0.0, max_spread_pct=1.0)).evaluate(
            candles=candles, rate=rate,
        )
        self.assertFalse(decision.passed)
        self.assertIn("wide spread", decision.reason)

    def test_skips_spread_gate_when_rate_missing(self) -> None:
        candles = _candles([100.0 + 0.5 * i for i in range(30)], hl_spread_pct=1.0)
        decision = ActivityFilter(_cfg(min_atr_pct=0.0)).evaluate(
            candles=candles, rate=None,
        )
        self.assertTrue(decision.passed)
        self.assertIsNone(decision.spread_pct)
        self.assertIn("spread=n/a", decision.reason)

    def test_skips_spread_gate_when_rate_missing_legs(self) -> None:
        candles = _candles([100.0 + 0.5 * i for i in range(30)], hl_spread_pct=1.0)
        decision = ActivityFilter(_cfg(min_atr_pct=0.0)).evaluate(
            candles=candles, rate=_rate(ask=None, bid=100.0),
        )
        self.assertTrue(decision.passed)
        self.assertIsNone(decision.spread_pct)

    def test_rejects_when_too_few_candles(self) -> None:
        candles = _candles([100.0, 101.0, 102.0], hl_spread_pct=1.0)
        decision = ActivityFilter(_cfg(activity_min_candles=20)).evaluate(
            candles=candles, rate=_rate(100.1, 100.0),
        )
        self.assertFalse(decision.passed)
        self.assertIn("insufficient candles", decision.reason)

    def test_rejects_inverted_spread(self) -> None:
        candles = _candles([100.0 + 0.5 * i for i in range(30)], hl_spread_pct=1.0)
        # ask < bid → treated as missing data (rate is unreliable); we
        # should not gate on it AND we should not blow up.
        decision = ActivityFilter(_cfg(min_atr_pct=0.0)).evaluate(
            candles=candles, rate=_rate(ask=99.0, bid=100.0),
        )
        self.assertTrue(decision.passed)
        self.assertIsNone(decision.spread_pct)

    def test_short_summary_includes_both_metrics(self) -> None:
        d = ActivityDecision(passed=True, reason="ok", atr_pct=0.5, spread_pct=0.1)
        self.assertEqual(d.short_summary(), "atr=0.50% spread=0.100%")


if __name__ == "__main__":
    unittest.main()
