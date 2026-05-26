"""Persistence for the performance tracker.

Three files, all under ``data/``:

- ``perf_open_positions.json`` — JSON dict mapping ``position_id -> OpenTradeState``.
  Loaded at startup so a restart doesn't forget MFE/MAE we've already
  recorded for currently-open bot trades. Written atomically.
- ``perf_closed_trades.jsonl`` — append-only ledger of every closed
  bot trade.
- ``perf_daily.jsonl`` — append-only one row per UTC day.

Each writer is guarded by a threading.Lock so concurrent calls from
the cycle thread and the controller thread don't tear a JSON file.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Iterable

from .types import DailySnapshot, OpenTradeState, RealizedTrade


class PerformanceStorage:
    """Encapsulates file IO for the performance tracker."""

    def __init__(
        self,
        data_dir: Path,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._open_path = self._dir / "perf_open_positions.json"
        self._closed_path = self._dir / "perf_closed_trades.jsonl"
        self._daily_path = self._dir / "perf_daily.jsonl"
        self._lock = threading.Lock()
        self._logger = logger or logging.getLogger("etrader.performance")

    # ------------------------------------------------------------------
    # Open positions snapshot — JSON dict
    # ------------------------------------------------------------------

    def load_open_positions(self) -> dict[int, OpenTradeState]:
        if not self._open_path.exists():
            return {}
        with self._lock:
            try:
                raw = self._open_path.read_text(encoding="utf-8")
            except OSError as exc:
                self._logger.warning("perf open-positions read failed: %s", exc)
                return {}
        if not raw.strip():
            return {}
        try:
            data = json.loads(raw)
        except ValueError as exc:
            self._logger.warning("perf open-positions JSON corrupt: %s", exc)
            return {}
        if not isinstance(data, dict):
            return {}
        out: dict[int, OpenTradeState] = {}
        for k, v in data.items():
            if not isinstance(v, dict):
                continue
            try:
                pid = int(k)
            except (TypeError, ValueError):
                continue
            try:
                out[pid] = OpenTradeState.from_dict(v)
            except (TypeError, ValueError):
                continue
        return out

    def save_open_positions(self, state: dict[int, OpenTradeState]) -> None:
        payload = {str(pid): s.to_dict() for pid, s in state.items()}
        body = json.dumps(payload, indent=2, default=str)
        with self._lock:
            _atomic_write(self._open_path, body)

    # ------------------------------------------------------------------
    # Closed trades — JSONL append-only
    # ------------------------------------------------------------------

    def append_closed_trade(self, trade: RealizedTrade) -> None:
        line = json.dumps(trade.to_dict(), default=str)
        with self._lock:
            try:
                with self._closed_path.open("a", encoding="utf-8") as fh:
                    fh.write(line + os.linesep)
            except OSError as exc:
                self._logger.warning("perf closed-trade append failed: %s", exc)

    def read_closed_trades(
        self, *, limit: int | None = None
    ) -> list[RealizedTrade]:
        return _read_jsonl(self._closed_path, RealizedTrade.from_dict, limit, self._logger)

    # ------------------------------------------------------------------
    # Daily snapshots — JSONL append-only
    # ------------------------------------------------------------------

    def append_daily(self, snap: DailySnapshot) -> None:
        line = json.dumps(snap.to_dict(), default=str)
        with self._lock:
            try:
                with self._daily_path.open("a", encoding="utf-8") as fh:
                    fh.write(line + os.linesep)
            except OSError as exc:
                self._logger.warning("perf daily-snapshot append failed: %s", exc)

    def read_dailies(
        self, *, limit: int | None = None
    ) -> list[DailySnapshot]:
        return _read_jsonl(self._daily_path, DailySnapshot.from_dict, limit, self._logger)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _atomic_write(path: Path, contents: str) -> None:
    """Write ``contents`` to ``path`` atomically (write-temp-then-rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".perf.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(contents)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_jsonl(
    path: Path,
    constructor,
    limit: int | None,
    logger: logging.Logger,
) -> list:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.warning("read %s failed: %s", path.name, exc)
        return []
    if limit is not None and limit > 0:
        lines = lines[-(limit * 2):]
    out = []
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
            out.append(constructor(obj))
        except (TypeError, ValueError):
            continue
    if limit is not None and limit > 0:
        out = out[-limit:]
    return out
