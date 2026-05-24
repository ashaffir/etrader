"""Wrapper around eToro's instrument feed posts endpoint.

The feed is heavyweight (text content, user metadata, reactions) so
we project it down to a small :class:`InstrumentFeedSummary` the
strategy layer can consume cheaply. We also implement TTL caching
because feed contents change far more slowly than a 60-second cycle.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping

from .client import EtoroClient


@dataclass(frozen=True)
class InstrumentFeedSummary:
    """Compact representation of recent feed posts for one instrument."""

    instrument_id: int
    post_count: int
    posts_24h: int
    bullish_keyword_count: int
    bearish_keyword_count: int
    fetched_at_unix: float
    sample_titles: tuple[str, ...]

    @property
    def is_stale(self) -> bool:
        return self.post_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "post_count": self.post_count,
            "posts_24h": self.posts_24h,
            "bullish_keywords": self.bullish_keyword_count,
            "bearish_keywords": self.bearish_keyword_count,
            "sample_titles": list(self.sample_titles),
        }


_BULLISH = (
    "buy", "long", "bull", "bullish", "moon", "breakout", "uptrend",
    "support", "accumulate", "pump",
)
_BEARISH = (
    "sell", "short", "bear", "bearish", "dump", "breakdown", "downtrend",
    "resistance", "distribute", "crash",
)


class InstrumentFeedFetcher:
    """Thread-safe TTL cache + eToro client wrapper."""

    def __init__(
        self,
        client: EtoroClient,
        *,
        take: int = 20,
        cache_ttl_seconds: float = 600.0,
    ) -> None:
        self._client = client
        self._take = max(1, min(int(take), 100))
        self._ttl = max(60.0, float(cache_ttl_seconds))
        self._cache: dict[int, InstrumentFeedSummary] = {}
        self._lock = threading.RLock()

    def fetch(self, instrument_id: int) -> InstrumentFeedSummary:
        now = time.time()
        with self._lock:
            cached = self._cache.get(instrument_id)
            if cached and now - cached.fetched_at_unix < self._ttl:
                return cached
        try:
            payload = self._client.get(
                f"/feeds/instrument/{instrument_id}",
                params={"take": self._take, "offset": 0},
                retries=1,
            )
        except Exception:  # noqa: BLE001 - any failure → return empty summary
            payload = None
        summary = _summarize_feed(instrument_id, payload, now)
        with self._lock:
            self._cache[instrument_id] = summary
        return summary

    def cached(self, instrument_id: int) -> InstrumentFeedSummary | None:
        with self._lock:
            return self._cache.get(instrument_id)


def _summarize_feed(
    instrument_id: int,
    payload: Mapping[str, Any] | None,
    fetched_at: float,
) -> InstrumentFeedSummary:
    if not payload:
        return InstrumentFeedSummary(
            instrument_id=instrument_id,
            post_count=0,
            posts_24h=0,
            bullish_keyword_count=0,
            bearish_keyword_count=0,
            fetched_at_unix=fetched_at,
            sample_titles=(),
        )

    posts: list[Mapping[str, Any]] = []
    for key in ("posts", "discussions", "items", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            posts = [p for p in value if isinstance(p, Mapping)]
            break

    bullish = bearish = 0
    posts_24h = 0
    sample_titles: list[str] = []
    cutoff = fetched_at - 86400.0

    for post in posts:
        text = " ".join(
            str(post.get(k) or "")
            for k in ("title", "content", "text", "body")
        ).lower()
        for kw in _BULLISH:
            if kw in text:
                bullish += 1
                break
        for kw in _BEARISH:
            if kw in text:
                bearish += 1
                break
        ts = _parse_ts(post.get("createdAt") or post.get("postDate") or post.get("timestamp"))
        if ts is not None and ts >= cutoff:
            posts_24h += 1
        title = str(post.get("title") or post.get("subject") or "").strip()
        if title and len(sample_titles) < 3:
            sample_titles.append(title[:80])

    return InstrumentFeedSummary(
        instrument_id=instrument_id,
        post_count=len(posts),
        posts_24h=posts_24h,
        bullish_keyword_count=bullish,
        bearish_keyword_count=bearish,
        fetched_at_unix=fetched_at,
        sample_titles=tuple(sample_titles),
    )


def _parse_ts(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) / (1000.0 if value > 1e12 else 1.0)
    if isinstance(value, str):
        try:
            from datetime import datetime
            stripped = value.replace("Z", "+00:00")
            return datetime.fromisoformat(stripped).timestamp()
        except ValueError:
            return None
    return None
