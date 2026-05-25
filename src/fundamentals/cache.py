"""Persisted fundamentals cache.

The cache is the single piece of fundamentals state the rest of the bot
talks to. It owns:

- A symbol → :class:`FundamentalsSnapshot` map, persisted as a single
  JSON document at ``data/fundamentals_cache.json``.
- A freshness policy: each entry is considered fresh for
  ``refresh_after_hours`` hours, and *always* refreshed once its
  recorded ``next_earnings_unix`` has passed (earnings reset
  fundamentals overnight).
- A small bookkeeping field per symbol (``next_attempt_unix``) so
  failed fetches don't hammer yfinance every cycle.

Concurrency
-----------
All mutating methods hold a :class:`threading.RLock`. The cache is
designed to be shared between the cycle (writer, on universe refresh)
and the HTTP control thread (reader, on ``/fundamentals``).

The cache itself never blocks the cycle — refresh calls are bounded by
``budget_per_refresh`` and the cycle's stop-event short-circuits the
batch so Ctrl-C is responsive even mid-refresh.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, Mapping

from .types import FundamentalsFetcher, FundamentalsSnapshot


DEFAULT_PATH = Path("data") / "fundamentals_cache.json"
DEFAULT_REFRESH_AFTER_HOURS = 24.0
DEFAULT_FAILURE_BACKOFF_HOURS = 6.0


class FundamentalsCache:
    """Symbol-keyed cache with freshness + earnings-aware refresh logic."""

    def __init__(
        self,
        *,
        fetcher: FundamentalsFetcher,
        path: Path = DEFAULT_PATH,
        refresh_after_hours: float = DEFAULT_REFRESH_AFTER_HOURS,
        failure_backoff_hours: float = DEFAULT_FAILURE_BACKOFF_HOURS,
        clock: Callable[[], float] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._fetcher = fetcher
        self._path = Path(path)
        # Floor both intervals at 5 minutes so a misconfigured 0 won't
        # turn the cache into an "every cycle, always" refresh storm.
        self._refresh_after = max(300.0, float(refresh_after_hours) * 3600.0)
        self._failure_backoff = max(300.0, float(failure_backoff_hours) * 3600.0)
        self._clock = clock or time.time
        self._log = logger or logging.getLogger("etrader.fundamentals.cache")
        self._lock = threading.RLock()
        self._items: dict[str, FundamentalsSnapshot] = {}
        # ``next_attempt_unix`` per symbol — used to back off after a
        # provider error so we don't hammer yfinance every cycle.
        self._next_attempt: dict[str, float] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self._log.warning("[fundamentals] cache load failed: %s", exc)
            return
        items = raw.get("items") if isinstance(raw, dict) else None
        if not isinstance(items, dict):
            return
        for sym, blob in items.items():
            if not isinstance(blob, Mapping):
                continue
            try:
                snap = FundamentalsSnapshot.from_dict(blob)
            except (TypeError, ValueError):
                continue
            self._items[snap.symbol.upper()] = snap

    def save(self) -> None:
        """Persist the cache atomically (write-and-rename)."""
        with self._lock:
            body = {"items": {sym: snap.to_dict() for sym, snap in self._items.items()}}
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self._path.with_suffix(self._path.suffix + ".tmp")
                tmp.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
                os.replace(tmp, self._path)
            except OSError as exc:
                self._log.warning("[fundamentals] cache save failed: %s", exc)

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------
    # Read paths
    # ------------------------------------------------------------------

    def get(self, symbol: str) -> FundamentalsSnapshot | None:
        sym = (symbol or "").strip().upper()
        if not sym:
            return None
        with self._lock:
            return self._items.get(sym)

    def all(self) -> dict[str, FundamentalsSnapshot]:
        with self._lock:
            return dict(self._items)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def __contains__(self, symbol: object) -> bool:
        if not isinstance(symbol, str):
            return False
        with self._lock:
            return symbol.strip().upper() in self._items

    # ------------------------------------------------------------------
    # Freshness policy
    # ------------------------------------------------------------------

    def is_stale(self, symbol: str) -> bool:
        """Return True if ``symbol`` should be re-fetched right now.

        Stale if any of these hold:
        - No entry yet for the symbol.
        - The entry is older than ``refresh_after_hours``.
        - The recorded next-earnings timestamp has already passed
          (earnings reset valuations / margins overnight).
        """
        sym = (symbol or "").strip().upper()
        if not sym:
            return False
        now = self._clock()
        with self._lock:
            snap = self._items.get(sym)
            if snap is None:
                return self._next_attempt.get(sym, 0.0) <= now
            if (now - snap.fetched_at_unix) >= self._refresh_after:
                return True
            if snap.next_earnings_unix and snap.next_earnings_unix < now:
                # Earnings just passed: the fundamentals payload is
                # almost certainly out of date.
                return True
            return False

    def needs_refresh(self, symbols: Iterable[str]) -> list[str]:
        """Return the subset of ``symbols`` whose entries should be re-fetched."""
        return [s for s in symbols if self.is_stale(s)]

    # ------------------------------------------------------------------
    # Write path (refresh batches)
    # ------------------------------------------------------------------

    def refresh(
        self,
        symbols: Iterable[str],
        *,
        budget: int | None = None,
        is_stopping: Callable[[], bool] | None = None,
    ) -> dict[str, str]:
        """Re-fetch up to ``budget`` stale symbols. Persists once at the end.

        ``is_stopping`` lets the caller plug in the global stop signal
        (same one the cycle uses) so a long batch can be cancelled
        mid-flight without leaving the cache in an inconsistent state.

        Returns a ``{symbol: status}`` map. Status is one of:
        - ``"refreshed"`` — fresh snapshot stored
        - ``"unchanged"`` — was still fresh, no refresh needed
        - ``"failed"``    — provider returned ``None`` or raised
        - ``"skipped"``   — outside the budget or stop requested
        """
        results: dict[str, str] = {}
        normalised = [s.strip().upper() for s in symbols if isinstance(s, str) and s.strip()]
        if not normalised:
            return results
        ordered_stale = [s for s in normalised if self.is_stale(s)]
        ordered_fresh = [s for s in normalised if s not in ordered_stale]
        for s in ordered_fresh:
            results[s] = "unchanged"

        cap = max(0, int(budget)) if budget is not None else len(ordered_stale)
        refresh_targets = ordered_stale[:cap]
        skipped = ordered_stale[cap:]
        for s in skipped:
            results[s] = "skipped"

        for sym in refresh_targets:
            if is_stopping is not None and is_stopping():
                results[sym] = "skipped"
                continue
            snap = self._fetcher.fetch(sym)
            now = self._clock()
            with self._lock:
                if snap is None:
                    self._next_attempt[sym] = now + self._failure_backoff
                    results[sym] = "failed"
                    continue
                self._items[sym] = snap
                self._next_attempt.pop(sym, None)
                results[sym] = "refreshed"

        # Persist once after the batch. Even partial progress is worth
        # saving; the next refresh will skip what we already covered.
        if any(v == "refreshed" for v in results.values()):
            self.save()
        return results
