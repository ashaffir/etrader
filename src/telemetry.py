"""Thread-safe snapshot store for the latest cycle's outputs.

The cycle runs in the main thread; the control HTTP server runs in a
daemon thread serving Telegram requests. Telegram needs read access to
the latest portfolio summary, the tracked universe, the most recent
decisions, the cycle counter, etc., without touching live eToro APIs
on every poll.

The :class:`TelemetryStore` provides those reads atomically. Writes
happen at well-known points in the cycle (top of cycle, after
portfolio fetch, after decisions). Readers always see a consistent
snapshot (we copy on read).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class _Mutable:
    cycle_count: int = 0
    last_cycle_started_unix: float | None = None
    last_cycle_finished_unix: float | None = None
    last_error: str | None = None

    portfolio_summary: dict[str, float] = field(default_factory=dict)
    portfolio_positions: list[dict[str, Any]] = field(default_factory=list)
    bot_owned_position_ids: list[int] = field(default_factory=list)

    tracked_symbols: list[str] = field(default_factory=list)
    tracked_instrument_ids: list[int] = field(default_factory=list)
    base_count: int = 0
    llm_count: int = 0

    last_decision_summary: str | None = None
    last_decision_llm_used: bool = False
    last_decision_actions: list[dict[str, Any]] = field(default_factory=list)


class TelemetryStore:
    """All-fields-protected store. Reads always return deep-ish copies."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data = _Mutable()

    # -- writes ---------------------------------------------------------

    def mark_cycle_started(self, cycle_count: int) -> None:
        with self._lock:
            self._data.cycle_count = cycle_count
            self._data.last_cycle_started_unix = time.time()
            self._data.last_error = None

    def mark_cycle_finished(self) -> None:
        with self._lock:
            self._data.last_cycle_finished_unix = time.time()

    def mark_cycle_error(self, message: str) -> None:
        with self._lock:
            self._data.last_error = message
            self._data.last_cycle_finished_unix = time.time()

    def update_portfolio(
        self,
        *,
        summary: Mapping[str, float],
        positions: list[dict[str, Any]],
        bot_owned_position_ids: list[int],
    ) -> None:
        with self._lock:
            self._data.portfolio_summary = dict(summary)
            self._data.portfolio_positions = [dict(p) for p in positions]
            self._data.bot_owned_position_ids = list(bot_owned_position_ids)

    def update_universe(
        self,
        *,
        instrument_ids: list[int],
        symbols: list[str],
        base_count: int,
        llm_count: int,
    ) -> None:
        with self._lock:
            self._data.tracked_instrument_ids = list(instrument_ids)
            self._data.tracked_symbols = list(symbols)
            self._data.base_count = int(base_count)
            self._data.llm_count = int(llm_count)

    def update_decision(
        self,
        *,
        summary: str | None,
        llm_used: bool,
        actions: list[dict[str, Any]],
    ) -> None:
        with self._lock:
            self._data.last_decision_summary = summary
            self._data.last_decision_llm_used = bool(llm_used)
            self._data.last_decision_actions = [dict(a) for a in actions]

    # -- reads ----------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            d = self._data
            return {
                "cycle_count": d.cycle_count,
                "last_cycle_started_unix": d.last_cycle_started_unix,
                "last_cycle_finished_unix": d.last_cycle_finished_unix,
                "last_error": d.last_error,
                "portfolio_summary": dict(d.portfolio_summary),
                "portfolio_positions": [dict(p) for p in d.portfolio_positions],
                "bot_owned_position_ids": list(d.bot_owned_position_ids),
                "tracked_symbols": list(d.tracked_symbols),
                "tracked_instrument_ids": list(d.tracked_instrument_ids),
                "base_count": d.base_count,
                "llm_count": d.llm_count,
                "last_decision_summary": d.last_decision_summary,
                "last_decision_llm_used": d.last_decision_llm_used,
                "last_decision_actions": [dict(a) for a in d.last_decision_actions],
            }
