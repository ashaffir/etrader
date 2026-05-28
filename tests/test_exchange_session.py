"""Per-exchange session tests.

Exercises every exchange the registry knows about with a small set of
"in-session" / "after-close" / "weekend" scenarios. The point is to
catch regressions where the bot silently falls back to NY hours for
non-US instruments — the original "WHY IS IT ONLY ACTIVE DURING US
HOURS" bug.

All UTC times below are computed against summer/winter dates to keep
DST behaviour in the test surface.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from datetime import datetime, timezone

from src.execution.exchange_session import (
    exchange_label,
    session_for,
    session_window_for,
)
from src.strategy.tools.base import AssetClass


def _utc(y: int, m: int, d: int, hh: int = 0, mm: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc)


@dataclass
class _FakeMeta:
    """Minimal stub matching :class:`InstrumentMeta` for these tests."""

    price_source: str


# ---------------------------------------------------------------------------
# Per-exchange open/close windows
# ---------------------------------------------------------------------------

class ExchangeWindowTests(unittest.TestCase):
    """Validate that each registered exchange opens at the right UTC
    instant on a representative summer weekday. The expectations
    encode local-time → UTC translation including DST.
    """

    DATE = (2026, 6, 17)  # Wednesday, EDT / BST / CEST / JST / HKT / AEST

    def _is_open(self, source: str, hour_utc: int, minute_utc: int = 0) -> bool:
        meta = _FakeMeta(price_source=source)
        now = _utc(*self.DATE, hh=hour_utc, mm=minute_utc)
        return session_for(meta, AssetClass.STOCK, now).is_open

    # -- US (EDT in June: open 13:30 UTC, close 20:00 UTC) -----------------
    def test_nasdaq_open_at_14_utc(self) -> None:
        self.assertTrue(self._is_open("nasdaq", 14))

    def test_nasdaq_closed_at_22_utc(self) -> None:
        self.assertFalse(self._is_open("nasdaq", 22))

    # -- UK (BST in June: LSE 08:00→16:30 London = 07:00→15:30 UTC) -------
    def test_lse_open_at_08_utc(self) -> None:
        self.assertTrue(self._is_open("lse", 8))

    def test_lse_closed_at_16_utc(self) -> None:
        self.assertFalse(self._is_open("lse", 16))

    def test_lse_open_when_us_is_closed(self) -> None:
        """The whole point of the refactor: LSE pre-NY-open works."""
        # 08:00 UTC = 09:00 London (open), 04:00 NY (pre-market).
        self.assertTrue(self._is_open("lse", 8))
        self.assertFalse(self._is_open("nasdaq", 8))

    # -- DE (CEST in June: XETRA 09:00→17:30 Berlin = 07:00→15:30 UTC) ----
    def test_xetra_open_at_10_utc(self) -> None:
        self.assertTrue(self._is_open("xetra", 10))

    def test_xetra_closed_at_16_utc(self) -> None:
        self.assertFalse(self._is_open("xetra", 16))

    # -- Japan (JST: TSE 09:00→15:30 Tokyo = 00:00→06:30 UTC) -------------
    def test_tse_open_at_03_utc(self) -> None:
        self.assertTrue(self._is_open("tse", 3))

    def test_tse_closed_at_08_utc(self) -> None:
        self.assertFalse(self._is_open("tse", 8))

    def test_tse_open_when_us_is_closed(self) -> None:
        # 03:00 UTC = midday Tokyo, 23:00 prev-day NY.
        self.assertTrue(self._is_open("tse", 3))
        self.assertFalse(self._is_open("nasdaq", 3))

    # -- Hong Kong (HKT: HKEX 09:30→16:00 = 01:30→08:00 UTC) -------------
    def test_hkex_open_at_05_utc(self) -> None:
        self.assertTrue(self._is_open("hkex", 5))

    def test_hkex_closed_at_10_utc(self) -> None:
        self.assertFalse(self._is_open("hkex", 10))

    # -- Australia (AEST: ASX 10:00→16:00 = 00:00→06:00 UTC) -------------
    def test_asx_open_at_03_utc(self) -> None:
        self.assertTrue(self._is_open("asx", 3))

    def test_asx_closed_at_08_utc(self) -> None:
        self.assertFalse(self._is_open("asx", 8))

    # -- Canada (EDT: TSX = NYSE hours, 13:30→20:00 UTC) -----------------
    def test_tsx_open_at_14_utc(self) -> None:
        self.assertTrue(self._is_open("tsx", 14))

    def test_tsx_closed_at_22_utc(self) -> None:
        self.assertFalse(self._is_open("tsx", 22))


# ---------------------------------------------------------------------------
# DST handling for LSE (Mar→Oct vs Oct→Mar)
# ---------------------------------------------------------------------------

class LseDstTests(unittest.TestCase):
    def test_lse_summer_open_at_08_utc(self) -> None:
        # June: London = UTC+1, so 08:00 UTC = 09:00 BST (in session).
        meta = _FakeMeta(price_source="lse")
        self.assertTrue(session_for(meta, AssetClass.STOCK, _utc(2026, 6, 17, 8)).is_open)

    def test_lse_winter_open_at_09_utc(self) -> None:
        # January: London = UTC, so 09:00 UTC = 09:00 GMT (in session).
        meta = _FakeMeta(price_source="lse")
        self.assertTrue(session_for(meta, AssetClass.STOCK, _utc(2026, 1, 14, 9)).is_open)

    def test_lse_winter_closed_at_17_utc(self) -> None:
        # 17:00 UTC = 17:00 GMT in winter — past 16:30 close.
        meta = _FakeMeta(price_source="lse")
        self.assertFalse(session_for(meta, AssetClass.STOCK, _utc(2026, 1, 14, 17)).is_open)


# ---------------------------------------------------------------------------
# Weekend + holiday behaviour
# ---------------------------------------------------------------------------

class WeekendTests(unittest.TestCase):
    def test_lse_closed_on_saturday(self) -> None:
        meta = _FakeMeta(price_source="lse")
        self.assertFalse(session_for(meta, AssetClass.STOCK, _utc(2026, 6, 20, 10)).is_open)

    def test_tse_closed_on_sunday(self) -> None:
        meta = _FakeMeta(price_source="tse")
        self.assertFalse(session_for(meta, AssetClass.STOCK, _utc(2026, 6, 21, 3)).is_open)


# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------

class FallbackTests(unittest.TestCase):
    def test_no_meta_falls_back_to_us_hours(self) -> None:
        # 14:00 UTC = 10:00 EDT → NY open.
        self.assertTrue(session_for(None, AssetClass.STOCK, _utc(2026, 6, 17, 14)).is_open)
        # 22:00 UTC = 18:00 EDT → past close.
        self.assertFalse(session_for(None, AssetClass.STOCK, _utc(2026, 6, 17, 22)).is_open)

    def test_unknown_price_source_falls_back_to_us(self) -> None:
        meta = _FakeMeta(price_source="kremlinex")
        self.assertTrue(
            session_for(meta, AssetClass.STOCK, _utc(2026, 6, 17, 14)).is_open
        )

    def test_crypto_always_open(self) -> None:
        for hour in (0, 8, 14, 22):
            self.assertTrue(
                session_for(None, AssetClass.CRYPTO, _utc(2026, 6, 21, hour)).is_open
            )

    def test_fx_closed_on_saturday(self) -> None:
        self.assertFalse(
            session_for(None, AssetClass.FX, _utc(2026, 6, 20, 10)).is_open
        )

    def test_fx_open_on_weekday(self) -> None:
        self.assertTrue(
            session_for(None, AssetClass.FX, _utc(2026, 6, 17, 10)).is_open
        )


# ---------------------------------------------------------------------------
# session_window_for + exchange_label
# ---------------------------------------------------------------------------

class WindowTests(unittest.TestCase):
    def test_lse_window_in_summer(self) -> None:
        meta = _FakeMeta(price_source="lse")
        open_utc, close_utc = session_window_for(
            meta, AssetClass.STOCK, _utc(2026, 6, 17, 10),
        )
        self.assertEqual(open_utc.hour, 7)   # 08:00 BST = 07:00 UTC
        self.assertEqual(close_utc.hour, 15)
        self.assertEqual(close_utc.minute, 30)

    def test_crypto_has_no_window(self) -> None:
        self.assertIsNone(
            session_window_for(None, AssetClass.CRYPTO, _utc(2026, 6, 17, 10))
        )

    def test_fx_has_no_window(self) -> None:
        self.assertIsNone(
            session_window_for(None, AssetClass.FX, _utc(2026, 6, 17, 10))
        )


class ExchangeLabelTests(unittest.TestCase):
    def test_lse_label(self) -> None:
        meta = _FakeMeta(price_source="lse")
        self.assertEqual(exchange_label(meta, AssetClass.STOCK), "LSE")

    def test_hkex_label(self) -> None:
        meta = _FakeMeta(price_source="hkex")
        self.assertEqual(exchange_label(meta, AssetClass.STOCK), "HKEX")

    def test_no_meta_label_is_nyse(self) -> None:
        self.assertEqual(exchange_label(None, AssetClass.STOCK), "NYSE")

    def test_crypto_label(self) -> None:
        self.assertEqual(exchange_label(None, AssetClass.CRYPTO), "CRYPTO")

    def test_fx_label(self) -> None:
        self.assertEqual(exchange_label(None, AssetClass.FX), "FX")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
