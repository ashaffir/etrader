"""Lower-level tests for the SQLite-backed :class:`PerformanceStorage`.

The high-level :class:`PerformanceTracker` tests cover the happy path,
but they don't exercise:

* one-shot import of legacy JSON/JSONL files,
* the daily upsert (replacing today's row in-place),
* concurrent reader/writer access (WAL + threading lock).

These tests pin those down so a future refactor can't regress them.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.performance.storage import PerformanceStorage
from src.performance.types import DailySnapshot, OpenTradeState, RealizedTrade


def _state(pid: int, pnl: float = 0.0) -> OpenTradeState:
    return OpenTradeState(
        position_id=pid,
        instrument_id=100 + pid,
        symbol=f"SYM{pid}",
        asset_class="stock",
        is_buy=True,
        amount_usd=100.0,
        units=1.0,
        open_rate=100.0,
        opened_at_iso="2026-05-26T13:00:00Z",
        last_pnl_usd=pnl,
    )


def _trade(pid: int, pnl: float, closed_at: str) -> RealizedTrade:
    return RealizedTrade(
        position_id=pid,
        instrument_id=100 + pid,
        symbol=f"SYM{pid}",
        asset_class="stock",
        is_buy=True,
        amount_usd=100.0,
        units=1.0,
        open_rate=100.0,
        close_rate=100.0 + pnl,
        realized_pnl_usd=pnl,
        realized_pnl_pct=pnl,
        opened_at_iso="2026-05-26T10:00:00Z",
        closed_at_iso=closed_at,
        hold_seconds=3600,
        mfe_usd=max(0.0, pnl),
        mae_usd=min(0.0, pnl),
        close_reason="reconciled",
    )


def _daily(date_iso: str, *, realized: float = 0.0) -> DailySnapshot:
    return DailySnapshot(
        date_iso=date_iso,
        equity_open=1000.0,
        equity_close=1000.0 + realized,
        equity_high=1010.0,
        equity_low=990.0,
        bot_realized_today_usd=realized,
        bot_trades_today=1,
    )


# ----------------------------------------------------------------------
# Schema + basic round trips
# ----------------------------------------------------------------------

class StorageBasicsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_creates_perf_sqlite_file(self) -> None:
        PerformanceStorage(self.dir).close()
        self.assertTrue((self.dir / "perf.sqlite").exists())

    def test_open_positions_round_trip(self) -> None:
        store = PerformanceStorage(self.dir)
        try:
            store.save_open_positions({1: _state(1, pnl=4.5), 2: _state(2, pnl=-1.2)})
            revived = store.load_open_positions()
            self.assertEqual(set(revived.keys()), {1, 2})
            self.assertAlmostEqual(revived[1].last_pnl_usd, 4.5)
        finally:
            store.close()

    def test_save_replaces_state_atomically(self) -> None:
        store = PerformanceStorage(self.dir)
        try:
            store.save_open_positions({1: _state(1), 2: _state(2)})
            store.save_open_positions({3: _state(3)})
            revived = store.load_open_positions()
            self.assertEqual(set(revived.keys()), {3})
        finally:
            store.close()

    def test_closed_trades_ordered_by_close_time(self) -> None:
        store = PerformanceStorage(self.dir)
        try:
            store.append_closed_trade(_trade(1, 1.0, "2026-05-26T15:00:00Z"))
            store.append_closed_trade(_trade(2, 2.0, "2026-05-26T13:00:00Z"))
            store.append_closed_trade(_trade(3, 3.0, "2026-05-26T14:00:00Z"))
            trades = store.read_closed_trades()
            self.assertEqual([t.position_id for t in trades], [2, 3, 1])
        finally:
            store.close()

    def test_closed_trades_limit_returns_most_recent(self) -> None:
        store = PerformanceStorage(self.dir)
        try:
            for i in range(10):
                store.append_closed_trade(
                    _trade(i, float(i), f"2026-05-26T{10 + i:02d}:00:00Z")
                )
            trades = store.read_closed_trades(limit=3)
            # Tail-of-chronological order with most-recent at the end.
            self.assertEqual([t.position_id for t in trades], [7, 8, 9])
        finally:
            store.close()


# ----------------------------------------------------------------------
# Daily upsert semantics
# ----------------------------------------------------------------------

class DailyUpsertTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.store = PerformanceStorage(self.dir)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def test_upsert_replaces_in_place(self) -> None:
        self.store.upsert_daily(_daily("2026-05-26", realized=1.0))
        self.store.upsert_daily(_daily("2026-05-26", realized=5.0))
        rows = self.store.read_dailies()
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0].bot_realized_today_usd, 5.0)

    def test_two_days_two_rows(self) -> None:
        self.store.upsert_daily(_daily("2026-05-26", realized=1.0))
        self.store.upsert_daily(_daily("2026-05-27", realized=2.0))
        rows = self.store.read_dailies()
        self.assertEqual([r.date_iso for r in rows], ["2026-05-26", "2026-05-27"])

    def test_append_daily_is_an_alias_for_upsert(self) -> None:
        self.store.append_daily(_daily("2026-05-26", realized=1.0))
        self.store.append_daily(_daily("2026-05-26", realized=2.0))
        rows = self.store.read_dailies()
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0].bot_realized_today_usd, 2.0)


# ----------------------------------------------------------------------
# Legacy JSON migration
# ----------------------------------------------------------------------

class LegacyMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _seed_legacy(self) -> None:
        (self.dir / "perf_open_positions.json").write_text(
            json.dumps({"7": _state(7, pnl=3.0).to_dict()}),
            encoding="utf-8",
        )
        (self.dir / "perf_closed_trades.jsonl").write_text(
            json.dumps(_trade(8, 4.0, "2026-05-26T14:00:00Z").to_dict()) + "\n"
            + json.dumps(_trade(9, -2.0, "2026-05-26T15:00:00Z").to_dict()) + "\n",
            encoding="utf-8",
        )
        (self.dir / "perf_daily.jsonl").write_text(
            json.dumps(_daily("2026-05-26", realized=2.0).to_dict()) + "\n",
            encoding="utf-8",
        )

    def test_migrates_all_three_legacy_files(self) -> None:
        self._seed_legacy()
        store = PerformanceStorage(self.dir)
        try:
            self.assertEqual(set(store.load_open_positions().keys()), {7})
            self.assertEqual(len(store.read_closed_trades()), 2)
            self.assertEqual(len(store.read_dailies()), 1)
        finally:
            store.close()
        # Legacy files renamed so we don't double-import on next boot.
        for name in (
            "perf_open_positions.json",
            "perf_closed_trades.jsonl",
            "perf_daily.jsonl",
        ):
            self.assertFalse((self.dir / name).exists(), name)
            self.assertTrue(
                any(p.suffix == ".legacy" for p in self.dir.glob(name + ".legacy"))
                or (self.dir / (name + ".legacy")).exists(),
                f"expected legacy rename for {name}",
            )

    def test_does_not_remigrate_after_sqlite_exists(self) -> None:
        self._seed_legacy()
        PerformanceStorage(self.dir).close()
        # Append fresh JSON that should NOT be re-imported because
        # perf.sqlite already exists.
        (self.dir / "perf_closed_trades.jsonl").write_text(
            json.dumps(_trade(100, 99.0, "2026-05-26T16:00:00Z").to_dict()) + "\n",
            encoding="utf-8",
        )
        store = PerformanceStorage(self.dir)
        try:
            self.assertEqual(len(store.read_closed_trades()), 2)  # not 3
        finally:
            store.close()

    def test_empty_dir_starts_clean(self) -> None:
        store = PerformanceStorage(self.dir)
        try:
            self.assertEqual(store.load_open_positions(), {})
            self.assertEqual(store.read_closed_trades(), [])
            self.assertEqual(store.read_dailies(), [])
        finally:
            store.close()


# ----------------------------------------------------------------------
# Concurrent read while writing — WAL must let the /stats reader in.
# ----------------------------------------------------------------------

class ConcurrentAccessTests(unittest.TestCase):
    """Smoke test that the WAL + threading.Lock setup doesn't deadlock
    when two threads hammer the store simultaneously."""

    def test_writer_and_reader_dont_deadlock(self) -> None:
        with TemporaryDirectory() as tmp:
            store = PerformanceStorage(Path(tmp))
            try:
                stop = threading.Event()

                def writer() -> None:
                    i = 0
                    while not stop.is_set():
                        store.save_open_positions({i: _state(i)})
                        i += 1

                def reader() -> int:
                    seen = 0
                    while not stop.is_set():
                        store.load_open_positions()
                        seen += 1
                    return seen

                w = threading.Thread(target=writer, daemon=True)
                r = threading.Thread(target=reader, daemon=True)
                w.start()
                r.start()
                time.sleep(0.15)
                stop.set()
                w.join(timeout=2.0)
                r.join(timeout=2.0)
                self.assertFalse(w.is_alive())
                self.assertFalse(r.is_alive())
            finally:
                store.close()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
