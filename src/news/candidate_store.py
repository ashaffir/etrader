"""Persisted, TTL'd store of news-driven universe candidates.

A :class:`Candidate` is a ticker that one or more news sources have
flagged as currently interesting, plus a human-readable reason and a
quality score. The store is the hand-off between the news pipeline
(producer) and the universe builder (consumer).

Persistence
-----------
Stored as a single JSON document under ``data/news_candidates.json``::

    {
      "candidates": {
        "AAPL": {
          "symbol": "AAPL",
          "score": 0.82,
          "sources": ["stocktwits", "yfinance"],
          "reason": "StockTwits trending #3 (+8K watchers); "
                    "yfinance: 'Apple beats Q3 estimates'",
          "headlines": ["StockTwits trending #3: $AAPL", "..."],
          "first_seen_unix": 1716580000.0,
          "last_seen_unix":  1716583600.0
        },
        ...
      }
    }

The store is intentionally lossy: when a candidate's ``last_seen_unix``
ages past the configurable TTL it is dropped on the next prune. The
universe builder is expected to re-rank after every prune, so stale
candidates don't linger in the active universe either.

Thread-safety
-------------
All mutating methods take an :class:`threading.RLock`. The store is
designed to be safe to share across the news aggregator (writer) and
the universe builder (reader) running in different threads.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Mapping


DEFAULT_PATH = Path("data") / "news_candidates.json"
DEFAULT_TTL_SECONDS = 24 * 60 * 60  # 24 h


@dataclass
class Candidate:
    """A single news-flagged ticker with all attached reasons.

    The store accumulates *reasons* — each ingestion bumps the score,
    extends the source set, and prepends the newest headline to the
    ring buffer. Headlines are capped at :data:`MAX_HEADLINES` to keep
    the JSON file from growing unbounded.
    """

    symbol: str
    score: float = 0.0
    sources: list[str] = field(default_factory=list)
    headlines: list[str] = field(default_factory=list)
    first_seen_unix: float = 0.0
    last_seen_unix: float = 0.0

    @property
    def reason(self) -> str:
        """Short, human-readable reason summary for UI / Telegram."""
        head = self.headlines[0] if self.headlines else f"{self.symbol} news"
        src = "+".join(self.sources) if self.sources else "?"
        return f"[{src}] {head}"


MAX_HEADLINES = 5


class CandidateStore:
    """Mutable, persisted, TTL'd map of ``symbol → Candidate``.

    Two paths in:

    * :meth:`record` — fold a new (symbol, source, headline, weight)
      observation into an existing candidate (or create it).
    * :meth:`extend` — bulk version, used by the aggregator.

    One path out:

    * :meth:`top` — return the highest-scoring N candidates whose
      ``last_seen_unix`` is within the TTL window.
    """

    def __init__(
        self,
        *,
        path: Path = DEFAULT_PATH,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        clock: "callable[[], float] | None" = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._path = path
        self._ttl = max(60.0, float(ttl_seconds))
        self._clock = clock or time.time
        self._log = logger or logging.getLogger("etrader.news.store")
        self._lock = threading.RLock()
        self._items: dict[str, Candidate] = {}
        self._load()

    # ----- persistence ---------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            self._log.warning("candidate store load failed: %s", exc)
            return
        body = raw.get("candidates") or {}
        for symbol, blob in body.items():
            if not isinstance(blob, Mapping):
                continue
            try:
                cand = Candidate(
                    symbol=str(blob.get("symbol") or symbol).upper(),
                    score=float(blob.get("score") or 0.0),
                    sources=[str(s) for s in (blob.get("sources") or []) if isinstance(s, str)],
                    headlines=[str(h) for h in (blob.get("headlines") or []) if isinstance(h, str)],
                    first_seen_unix=float(blob.get("first_seen_unix") or 0.0),
                    last_seen_unix=float(blob.get("last_seen_unix") or 0.0),
                )
            except (TypeError, ValueError):
                continue
            self._items[cand.symbol] = cand

    def save(self) -> None:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            body = {
                "candidates": {sym: asdict(c) for sym, c in self._items.items()},
            }
            try:
                self._path.write_text(
                    json.dumps(body, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            except OSError as exc:
                self._log.warning("candidate store save failed: %s", exc)

    # ----- mutation ------------------------------------------------------

    def record(
        self,
        *,
        symbol: str,
        source: str,
        headline: str,
        weight: float = 1.0,
    ) -> Candidate:
        """Fold a single (symbol, source, headline) observation in.

        Idempotent on ``(symbol, source)``: the source is added to the
        list only once, but the score is bumped every call (capped by
        the weight). ``headline`` is prepended (most recent first) and
        the ring buffer is trimmed to :data:`MAX_HEADLINES`.
        """
        sym = symbol.strip().upper()
        if not sym:
            raise ValueError("symbol must be non-empty")
        now = self._clock()
        with self._lock:
            cand = self._items.get(sym)
            if cand is None:
                cand = Candidate(symbol=sym, first_seen_unix=now)
                self._items[sym] = cand
            if source and source not in cand.sources:
                cand.sources.append(source)
            cand.score += max(0.0, float(weight))
            if headline:
                clean = headline.strip()
                if clean and clean not in cand.headlines:
                    cand.headlines.insert(0, clean)
                    if len(cand.headlines) > MAX_HEADLINES:
                        cand.headlines = cand.headlines[:MAX_HEADLINES]
            cand.last_seen_unix = now
            return cand

    def extend(self, observations: Iterable[Mapping[str, object]]) -> int:
        """Bulk-record observations.

        Each item must contain at least ``symbol``, ``source`` and
        ``headline``; ``weight`` is optional (default 1.0). Returns the
        number of observations applied.
        """
        applied = 0
        for obs in observations:
            symbol = obs.get("symbol")
            source = obs.get("source")
            headline = obs.get("headline")
            if not isinstance(symbol, str) or not isinstance(source, str):
                continue
            weight_raw = obs.get("weight", 1.0)
            try:
                weight = float(weight_raw) if weight_raw is not None else 1.0
            except (TypeError, ValueError):
                weight = 1.0
            self.record(
                symbol=symbol,
                source=source,
                headline=str(headline or ""),
                weight=weight,
            )
            applied += 1
        return applied

    def prune(self) -> int:
        """Drop candidates whose ``last_seen_unix`` is older than the TTL.

        Returns the number of removed entries. Cheap — O(n) sweep.
        """
        cutoff = self._clock() - self._ttl
        removed = 0
        with self._lock:
            for sym in list(self._items.keys()):
                if self._items[sym].last_seen_unix < cutoff:
                    del self._items[sym]
                    removed += 1
        return removed

    def clear(self) -> None:
        """Drop every candidate. Useful for tests."""
        with self._lock:
            self._items.clear()

    # ----- query ---------------------------------------------------------

    def top(self, n: int | None = None) -> list[Candidate]:
        """Return live candidates sorted by descending score.

        TTL pruning is *not* applied here — call :meth:`prune` first if
        the caller needs strict TTL semantics. Two candidates with the
        same score are tie-broken by ``last_seen_unix`` (more recent
        first) and then by symbol.
        """
        with self._lock:
            ordered = sorted(
                self._items.values(),
                key=lambda c: (-c.score, -c.last_seen_unix, c.symbol),
            )
        if n is None:
            return list(ordered)
        return list(ordered[: max(0, int(n))])

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def __contains__(self, symbol: object) -> bool:
        if not isinstance(symbol, str):
            return False
        with self._lock:
            return symbol.strip().upper() in self._items

    def __iter__(self) -> Iterator[Candidate]:
        with self._lock:
            snapshot = list(self._items.values())
        return iter(snapshot)
