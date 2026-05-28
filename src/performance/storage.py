"""SQLite-backed persistence for the performance tracker.

Why SQLite instead of three JSON/JSONL files?

* **Atomic writes** — every save is wrapped in a transaction, so a
  crash mid-write never leaves a torn file.  The previous JSON
  layout rewrote the entire ``perf_open_positions.json`` on every
  cycle and could corrupt on a power loss.
* **No whole-file rewrites** — open-position bookkeeping that used
  to rewrite the entire file every cycle now writes only the deltas
  via prepared statements.
* **Concurrent reads while writes happen** — WAL mode lets the
  Telegram ``/stats`` handler read snapshots while the cycle thread
  is updating MFE/MAE without blocking.
* **Real upserts for the daily roller** — ``UPSERT`` semantics
  replace the previous "rewrite the entire JSONL minus the last
  line" hack.

Schema (three tables, primary keys keep things narrow):

* ``open_positions``  — current bot-owned positions (in-progress)
* ``closed_trades``   — append-only realized-trade ledger
* ``daily_snapshots`` — one row per UTC day (upsertable)

The schema, prepared statements and row converters live in
:mod:`._sqlite`. The one-shot legacy migration lives in
:mod:`._legacy`. This file keeps just the public class.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

from . import _legacy, _sqlite as _sql
from .types import DailySnapshot, OpenTradeState, RealizedTrade


class PerformanceStorage:
    """SQLite-backed implementation of the perf store.

    Public API is identical to the previous file-based implementation
    so callers don't change. ``data_dir`` is the directory that holds
    ``perf.sqlite`` (and is also scanned for legacy JSON/JSONL files
    on first run).
    """

    def __init__(
        self,
        data_dir: Path,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._dir / "perf.sqlite"
        self._logger = logger or logging.getLogger("etrader.performance")
        self._lock = threading.Lock()
        is_fresh = not self._db_path.exists()
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            # autocommit — we manage transactions explicitly via `with self._conn`
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._init_pragmas()
        self._apply_schema()
        if is_fresh:
            _legacy.migrate(
                conn=self._conn, data_dir=self._dir, logger=self._logger,
            )

    def close(self) -> None:
        """Close the underlying SQLite connection — call on shutdown."""
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass

    # ------------------------------------------------------------------
    # Open positions — bulk replace (the cycle hands us the full picture)
    # ------------------------------------------------------------------

    def load_open_positions(self) -> dict[int, OpenTradeState]:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM open_positions")
            rows = cur.fetchall()
        out: dict[int, OpenTradeState] = {}
        for row in rows:
            try:
                state = _sql.row_to_open_state(row)
            except (TypeError, ValueError) as exc:
                self._logger.warning("perf open_positions row corrupt: %s", exc)
                continue
            out[state.position_id] = state
        return out

    def save_open_positions(self, state: dict[int, OpenTradeState]) -> None:
        rows = [_sql.open_state_to_row(s) for s in state.values()]
        with self._lock:
            try:
                with self._conn:  # BEGIN…COMMIT
                    self._conn.execute("DELETE FROM open_positions")
                    if rows:
                        self._conn.executemany(_sql.INSERT_OPEN, rows)
            except sqlite3.Error as exc:
                self._logger.warning(
                    "perf open_positions save failed: %s", exc
                )

    # ------------------------------------------------------------------
    # Closed-trade ledger — append-only
    # ------------------------------------------------------------------

    def append_closed_trade(self, trade: RealizedTrade) -> None:
        row = _sql.closed_trade_to_row(trade)
        with self._lock:
            try:
                # INSERT OR REPLACE protects against a duplicate
                # position_id (e.g. cycle retries record_close on
                # the same pid). The ledger then carries the latest
                # observation, which is safer than a stale row.
                self._conn.execute(_sql.INSERT_OR_REPLACE_CLOSED, row)
            except sqlite3.Error as exc:
                self._logger.warning(
                    "perf closed_trade append failed: %s", exc
                )

    def read_closed_trades(
        self, *, limit: int | None = None
    ) -> list[RealizedTrade]:
        # Ordered chronologically so callers can slice the tail with
        # ``[-N:]`` the same way they did against the JSONL ledger.
        sql = "SELECT * FROM closed_trades ORDER BY closed_at_iso ASC, position_id ASC"
        params: tuple = ()
        if limit is not None and limit > 0:
            sql = (
                "SELECT * FROM ("
                "SELECT * FROM closed_trades "
                "ORDER BY closed_at_iso DESC, position_id DESC LIMIT ?"
                ") ORDER BY closed_at_iso ASC, position_id ASC"
            )
            params = (int(limit),)
        with self._lock:
            cur = self._conn.execute(sql, params)
            rows = cur.fetchall()
        out: list[RealizedTrade] = []
        for row in rows:
            try:
                out.append(_sql.row_to_closed_trade(row))
            except (TypeError, ValueError) as exc:
                self._logger.warning("perf closed_trades row corrupt: %s", exc)
        return out

    # ------------------------------------------------------------------
    # Daily snapshots — upsert per UTC date
    # ------------------------------------------------------------------

    def append_daily(self, snap: DailySnapshot) -> None:
        """Old API name — preserved as an alias for :meth:`upsert_daily`.

        Under SQLite this is functionally an upsert because the date
        is the primary key. The :class:`DailyRoller` now calls
        :meth:`upsert_daily` directly; this remains for migration
        replay and any external callers.
        """
        self.upsert_daily(snap)

    def upsert_daily(self, snap: DailySnapshot) -> None:
        row = _sql.daily_to_row(snap)
        with self._lock:
            try:
                self._conn.execute(_sql.UPSERT_DAILY, row)
            except sqlite3.Error as exc:
                self._logger.warning(
                    "perf daily_snapshots upsert failed: %s", exc
                )

    def read_dailies(
        self, *, limit: int | None = None
    ) -> list[DailySnapshot]:
        sql = "SELECT * FROM daily_snapshots ORDER BY date_iso ASC"
        params: tuple = ()
        if limit is not None and limit > 0:
            sql = (
                "SELECT * FROM ("
                "SELECT * FROM daily_snapshots ORDER BY date_iso DESC LIMIT ?"
                ") ORDER BY date_iso ASC"
            )
            params = (int(limit),)
        with self._lock:
            cur = self._conn.execute(sql, params)
            rows = cur.fetchall()
        out: list[DailySnapshot] = []
        for row in rows:
            try:
                out.append(_sql.row_to_daily(row))
            except (TypeError, ValueError) as exc:
                self._logger.warning("perf daily_snapshots row corrupt: %s", exc)
        return out

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _init_pragmas(self) -> None:
        # WAL = concurrent reads + writer; synchronous=NORMAL is the
        # WAL-recommended durability/perf trade-off.
        try:
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = NORMAL")
            self._conn.execute("PRAGMA foreign_keys = ON")
        except sqlite3.Error as exc:
            self._logger.warning("perf sqlite pragma failed: %s", exc)

    def _apply_schema(self) -> None:
        try:
            with self._conn:
                self._conn.executescript(_sql.SCHEMA)
        except sqlite3.Error as exc:
            self._logger.error("perf sqlite schema failed: %s", exc)
            raise
