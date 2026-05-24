"""Append-only trade-history log (JSONL).

Why JSONL: each line is a self-contained JSON object so the file can
grow indefinitely without a parser ever needing to load it whole. We
also tolerate a single corrupted line — the loader skips bad rows.

The Telegram service reads this file (via the control HTTP API) to
answer ``/history``. The trading bot writes one entry per
ExecutionResult emitted by the executor.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class TradeHistoryEntry:
    timestamp: str        # ISO-8601 UTC, second resolution
    action: str           # "BUY" | "CLOSE"
    status: str           # "ok" | "ambiguous" | "failed" | "skipped" | "rate_limited" | "panic_close"
    symbol: str
    instrument_id: int | None
    amount_usd: float | None
    order_id: int | None
    position_id: int | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "action": self.action,
            "status": self.status,
            "symbol": self.symbol,
            "instrument_id": self.instrument_id,
            "amount_usd": self.amount_usd,
            "order_id": self.order_id,
            "position_id": self.position_id,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TradeHistoryEntry":
        return cls(
            timestamp=str(data.get("timestamp") or ""),
            action=str(data.get("action") or ""),
            status=str(data.get("status") or ""),
            symbol=str(data.get("symbol") or ""),
            instrument_id=_opt_int(data.get("instrument_id")),
            amount_usd=_opt_float(data.get("amount_usd")),
            order_id=_opt_int(data.get("order_id")),
            position_id=_opt_int(data.get("position_id")),
            detail=str(data.get("detail") or ""),
        )


class TradeHistoryLog:
    """Thread-safe writer + reader for a single JSONL trade-history file."""

    def __init__(
        self,
        path: Path,
        *,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
    ) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._logger = logger or logging.getLogger("etrader.trade_history")

    @property
    def path(self) -> Path:
        return self._path

    def append(self, entry: TradeHistoryEntry) -> None:
        line = json.dumps(entry.to_dict(), default=str)
        with self._lock:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(line + os.linesep)
            except OSError as exc:
                self._logger.warning("trade history append failed: %s", exc)

    def append_many(self, entries: Iterable[TradeHistoryEntry]) -> None:
        for e in entries:
            self.append(e)

    def tail(self, *, limit: int = 50) -> list[TradeHistoryEntry]:
        """Return at most ``limit`` most-recent entries (newest last).

        Reads the whole file. Acceptable because we cap the file size
        externally (rotated alongside ``logs/trader.log``) and Telegram
        replies use ``limit <= 50``.
        """
        if not self._path.exists() or limit <= 0:
            return []
        with self._lock:
            try:
                lines = self._path.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                self._logger.warning("trade history read failed: %s", exc)
                return []
        out: list[TradeHistoryEntry] = []
        for raw in lines[-limit * 2:]:  # pad to absorb skipped rows
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
                out.append(TradeHistoryEntry.from_dict(obj))
            except (TypeError, ValueError):
                continue
        return out[-limit:]


# ---------------------------------------------------------------------------
# Helpers — used by callers building entries from ExecutionResult
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _opt_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _opt_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
