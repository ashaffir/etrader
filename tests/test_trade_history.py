"""Tests for :class:`src.trade_history.TradeHistoryLog`."""

import tempfile
import unittest
from pathlib import Path

from src.trade_history import TradeHistoryEntry, TradeHistoryLog, utc_now_iso


class TradeHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "history.jsonl"

    def _entry(self, **overrides) -> TradeHistoryEntry:
        defaults = dict(
            timestamp=utc_now_iso(),
            action="BUY",
            status="ok",
            symbol="AAPL",
            instrument_id=1,
            amount_usd=250.0,
            order_id=99,
            position_id=None,
            detail="orderID=99",
        )
        defaults.update(overrides)
        return TradeHistoryEntry(**defaults)

    def test_append_and_tail(self) -> None:
        log = TradeHistoryLog(self.path)
        log.append(self._entry(symbol="AAPL"))
        log.append(self._entry(symbol="MSFT"))
        log.append(self._entry(symbol="NVDA"))

        all_entries = log.tail(limit=10)
        self.assertEqual([e.symbol for e in all_entries], ["AAPL", "MSFT", "NVDA"])

    def test_tail_respects_limit(self) -> None:
        log = TradeHistoryLog(self.path)
        for sym in ("A", "B", "C", "D", "E"):
            log.append(self._entry(symbol=sym))
        last_two = log.tail(limit=2)
        self.assertEqual([e.symbol for e in last_two], ["D", "E"])

    def test_tail_skips_corrupted_lines(self) -> None:
        log = TradeHistoryLog(self.path)
        log.append(self._entry(symbol="AAPL"))
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write("garbage line\n")
        log.append(self._entry(symbol="MSFT"))

        out = log.tail(limit=10)
        self.assertEqual([e.symbol for e in out], ["AAPL", "MSFT"])

    def test_missing_file_returns_empty(self) -> None:
        log = TradeHistoryLog(self.path)
        self.assertEqual(log.tail(limit=5), [])

    def test_round_trips_optional_fields(self) -> None:
        log = TradeHistoryLog(self.path)
        log.append(self._entry(amount_usd=None, order_id=None, position_id=42))
        loaded = log.tail(limit=1)[0]
        self.assertIsNone(loaded.amount_usd)
        self.assertIsNone(loaded.order_id)
        self.assertEqual(loaded.position_id, 42)


if __name__ == "__main__":
    unittest.main()
