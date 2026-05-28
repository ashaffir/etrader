"""One-shot JSON → SQLite import for the previous file-based perf store.

Called by :class:`PerformanceStorage` only when the SQLite file is
created fresh and legacy ``perf_*.json`` / ``.jsonl`` files exist in
the same directory. After import the source files are renamed with a
``.legacy`` suffix so a future boot won't double-import them.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Callable, Iterable

from . import _sqlite as _sql
from .types import DailySnapshot, OpenTradeState, RealizedTrade


LEGACY_FILES = (
    "perf_open_positions.json",
    "perf_closed_trades.jsonl",
    "perf_daily.jsonl",
)


def migrate(
    *,
    conn: sqlite3.Connection,
    data_dir: Path,
    logger: logging.Logger,
) -> int:
    """Import all three legacy files into ``conn``. Returns row count."""
    open_path = data_dir / "perf_open_positions.json"
    closed_path = data_dir / "perf_closed_trades.jsonl"
    daily_path = data_dir / "perf_daily.jsonl"
    imported = 0
    with conn:
        imported += _import_open(conn, open_path)
        imported += _import_jsonl(
            conn,
            closed_path,
            RealizedTrade.from_dict,
            _sql.INSERT_OR_REPLACE_CLOSED,
            _sql.closed_trade_to_row,
        )
        imported += _import_jsonl(
            conn,
            daily_path,
            DailySnapshot.from_dict,
            _sql.UPSERT_DAILY,
            _sql.daily_to_row,
        )
    if imported:
        logger.info(
            "performance store: migrated %s legacy rows from JSON → SQLite",
            imported,
        )
    _rename_legacy(data_dir, logger)
    return imported


def _import_open(conn: sqlite3.Connection, path: Path) -> int:
    if not path.exists():
        return 0
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return 0
    try:
        data = json.loads(raw) if raw.strip() else {}
    except ValueError:
        return 0
    if not isinstance(data, dict):
        return 0
    rows = []
    for _k, v in data.items():
        if not isinstance(v, dict):
            continue
        try:
            state = OpenTradeState.from_dict(v)
        except (TypeError, ValueError):
            continue
        rows.append(_sql.open_state_to_row(state))
    if rows:
        conn.executemany(_sql.INSERT_OPEN, rows)
    return len(rows)


def _import_jsonl(
    conn: sqlite3.Connection,
    path: Path,
    constructor: Callable[[dict], object],
    insert_sql: str,
    to_row: Callable[[object], tuple],
) -> int:
    rows = list(_iter_jsonl(path, constructor))
    if rows:
        conn.executemany(insert_sql, [to_row(r) for r in rows])
    return len(rows)


def _iter_jsonl(path: Path, constructor: Callable[[dict], object]) -> Iterable:
    if not path.exists():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        try:
            yield constructor(obj)
        except (TypeError, ValueError):
            continue


def _rename_legacy(data_dir: Path, logger: logging.Logger) -> None:
    for name in LEGACY_FILES:
        src = data_dir / name
        if not src.exists():
            continue
        dst = src.with_suffix(src.suffix + ".legacy")
        try:
            src.replace(dst)
        except OSError as exc:
            logger.warning(
                "perf legacy rename %s failed: %s", src.name, exc
            )
