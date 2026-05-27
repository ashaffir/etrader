"""LLM token-usage tracker — daily rollups + per-call-type breakdown.

Every chat-completion call routed through :class:`AzureFoundryClient`
hands its ``usage`` object to :meth:`LLMUsageTracker.record`. The
tracker:

* aggregates running daily totals in memory (so a hot /tokens query
  is cheap), and
* appends one JSONL line per UTC day to ``data/llm_usage.jsonl`` so
  long-term cost trends survive restarts.

A "call_type" string is also recorded with each entry (e.g.
``"decision"``, ``"qa"``, ``"universe_rotation"``) so the /tokens UI
can attribute spend to the user journey that incurred it.

This module is dependency-free other than the small price table in
:mod:`src.ai.pricing`. All file I/O is best-effort: if writing the
JSONL file fails (disk full / permission) the in-memory counters
still update, the tracker logs once, and life continues.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .pricing import TokenRates, lookup_rates, price_table_as_of


# Per-day file name suffix — kept ISO-8601 for grep-friendliness.
_DATE_FMT = "%Y-%m-%d"


@dataclass
class UsageEntry:
    """One chat-completion round's accounting."""

    timestamp_iso: str
    date_iso: str
    deployment: str
    call_type: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _WindowAccumulator:
    """Mutable counters for a rolling window (today, 7d, 30d, all-time)."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, entry: UsageEntry) -> None:
        self.calls += 1
        self.prompt_tokens += int(entry.prompt_tokens)
        self.completion_tokens += int(entry.completion_tokens)
        self.cached_tokens += int(entry.cached_tokens)
        self.cost_usd += float(entry.cost_usd)

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


class LLMUsageTracker:
    """Thread-safe LLM usage + cost ledger.

    Construct once at boot; pass to :class:`AzureFoundryClient` (or
    call :meth:`record` from anywhere that fires an LLM round-trip).

    The default rate lookup uses the deployment name on the client.
    ``deployment_override`` is a test hook to pin the rates table
    regardless of the client config.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        deployment: str | None = None,
        rates_override: TokenRates | None = None,
        keep_recent_n: int = 5_000,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
    ) -> None:
        self._path = Path(path)
        self._deployment = deployment or ""
        self._rates = rates_override or lookup_rates(deployment)
        self._lock = threading.Lock()
        self._recent: list[UsageEntry] = []
        self._keep_recent_n = int(keep_recent_n)
        self._by_day: dict[str, _WindowAccumulator] = defaultdict(_WindowAccumulator)
        self._by_call_type_today: dict[str, _WindowAccumulator] = defaultdict(
            _WindowAccumulator,
        )
        self._last_entry: UsageEntry | None = None
        self._today_iso = _today_utc()
        self._log = logger or logging.getLogger("etrader.ai.usage_tracker")
        self._restore()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
        call_type: str = "unknown",
        latency_ms: int | None = None,
        timestamp: datetime | None = None,
    ) -> UsageEntry:
        """Capture one chat-completion round's usage.

        Returns the :class:`UsageEntry` so the caller can log it.
        """
        ts = timestamp or datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        date_iso = ts.strftime(_DATE_FMT)
        total = int(prompt_tokens) + int(completion_tokens)
        cost = (
            self._rates.cost_for(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
            )
            if self._rates is not None else 0.0
        )
        entry = UsageEntry(
            timestamp_iso=ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            date_iso=date_iso,
            deployment=self._deployment,
            call_type=str(call_type or "unknown"),
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            cached_tokens=int(cached_tokens),
            total_tokens=total,
            cost_usd=cost,
            latency_ms=int(latency_ms) if latency_ms is not None else None,
        )

        with self._lock:
            if date_iso != self._today_iso:
                # Day rolled over — reset the "today" call-type bucket
                # so /tokens shows a clean slate for the new day.
                self._today_iso = date_iso
                self._by_call_type_today = defaultdict(_WindowAccumulator)
            self._by_day[date_iso].add(entry)
            self._by_call_type_today[entry.call_type].add(entry)
            self._recent.append(entry)
            if len(self._recent) > self._keep_recent_n:
                self._recent = self._recent[-self._keep_recent_n:]
            self._last_entry = entry
        self._append_jsonl(entry)
        return entry

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            today = self._today_iso
            return {
                "deployment": self._deployment,
                "rates": self._rates.to_dict() if self._rates is not None else None,
                "rates_as_of": price_table_as_of(),
                "today": self._window(days=1, today=today).to_dict(),
                "last_7d": self._window(days=7, today=today).to_dict(),
                "last_30d": self._window(days=30, today=today).to_dict(),
                "all_time": self._window(days=None, today=today).to_dict(),
                "by_call_type": {
                    k: acc.to_dict() for k, acc in self._by_call_type_today.items()
                },
                "last_call": (
                    self._last_entry.to_dict() if self._last_entry else None
                ),
                "recent_count": len(self._recent),
            }

    def _window(self, *, days: int | None, today: str) -> _WindowAccumulator:
        if days is None:
            agg = _WindowAccumulator()
            for acc in self._by_day.values():
                agg.calls += acc.calls
                agg.prompt_tokens += acc.prompt_tokens
                agg.completion_tokens += acc.completion_tokens
                agg.cached_tokens += acc.cached_tokens
                agg.cost_usd += acc.cost_usd
            return agg
        cutoff = (
            datetime.strptime(today, _DATE_FMT) - timedelta(days=days - 1)
        ).strftime(_DATE_FMT)
        agg = _WindowAccumulator()
        for date_iso, acc in self._by_day.items():
            if date_iso >= cutoff:
                agg.calls += acc.calls
                agg.prompt_tokens += acc.prompt_tokens
                agg.completion_tokens += acc.completion_tokens
                agg.cached_tokens += acc.cached_tokens
                agg.cost_usd += acc.cost_usd
        return agg

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _append_jsonl(self, entry: UsageEntry) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(entry.to_dict()) + "\n")
        except OSError as exc:
            self._log.warning("[usage] persist failed: %s", exc)

    def _restore(self) -> None:
        """Rebuild day-level totals from the JSONL on disk.

        Only fields we need for the windows are summed — we don't
        replay individual entries beyond keeping the last
        ``keep_recent_n`` so /tokens reflects the long-term ledger
        right after boot.
        """
        if not self._path.exists():
            return
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            self._log.warning("[usage] restore read failed: %s", exc)
            return
        recent: list[UsageEntry] = []
        for raw in lines:
            if not raw.strip():
                continue
            try:
                d = json.loads(raw)
            except (ValueError, TypeError):
                continue
            date_iso = str(d.get("date_iso") or "")
            if not date_iso:
                continue
            entry = UsageEntry(
                timestamp_iso=str(d.get("timestamp_iso") or ""),
                date_iso=date_iso,
                deployment=str(d.get("deployment") or ""),
                call_type=str(d.get("call_type") or "unknown"),
                prompt_tokens=int(d.get("prompt_tokens") or 0),
                completion_tokens=int(d.get("completion_tokens") or 0),
                cached_tokens=int(d.get("cached_tokens") or 0),
                total_tokens=int(d.get("total_tokens") or 0),
                cost_usd=float(d.get("cost_usd") or 0.0),
                latency_ms=(
                    int(d["latency_ms"]) if d.get("latency_ms") is not None else None
                ),
            )
            self._by_day[date_iso].add(entry)
            if date_iso == self._today_iso:
                self._by_call_type_today[entry.call_type].add(entry)
            recent.append(entry)
        if recent:
            self._recent = recent[-self._keep_recent_n:]
            self._last_entry = recent[-1]


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime(_DATE_FMT)


__all__ = [
    "LLMUsageTracker",
    "UsageEntry",
]
