"""Fan-out across news sources, extract tickers, score and store.

The aggregator is the one-and-only orchestration layer in the news
pipeline. It owns:

* a list of :class:`~src.news.sources.base.NewsSource` plug-ins;
* a :class:`~src.news.ticker_extractor.TickerExtractor` for prose
  headlines that don't ship tickers themselves;
* a :class:`~src.news.candidate_store.CandidateStore` for persisted,
  TTL'd output.

Per source it:

1. calls :meth:`NewsSource.fetch(since, known_symbols)`;
2. dedupes items by ``(source, url)``;
3. extracts tickers for items without explicit ``symbols``;
4. computes a per-observation weight from the source's configured
   base weight + freshness decay + sentiment magnitude;
5. records every (symbol, source, headline, weight) tuple into the
   candidate store.

Sources fail soft: a single broken source never aborts the whole run.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .candidate_store import CandidateStore
from .sources.base import NewsItem, NewsSource
from .ticker_extractor import TickerExtractor


# Default per-source weight multipliers. Tuned conservatively: discovery
# sources (StockTwits, SEC EDGAR) score higher than aggregator-style
# sources (Google News) because each item there is already pre-filtered
# by the source itself.
DEFAULT_SOURCE_WEIGHTS: Mapping[str, float] = {
    "stocktwits": 1.0,
    "sec_edgar": 1.2,
    "yfinance": 0.9,
    "yahoo_rss": 0.7,
    "google_news": 0.5,
}


# How fast freshness decays. With ``half_life_seconds = 21600`` (6 h)
# an item from 6 hours ago contributes half the weight of "just now".
DEFAULT_HALF_LIFE_SECONDS = 6 * 60 * 60


@dataclass
class AggregatorRunStats:
    """One run's bookkeeping — useful for ``/news`` and logs."""

    started_at_unix: float = 0.0
    finished_at_unix: float = 0.0
    items_fetched: int = 0
    items_kept: int = 0  # post-dedup
    observations_recorded: int = 0
    per_source_counts: dict[str, int] = field(default_factory=dict)
    per_source_errors: dict[str, str] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.finished_at_unix - self.started_at_unix)


class NewsAggregator:
    """Run all configured sources and update the candidate store."""

    def __init__(
        self,
        sources: Sequence[NewsSource],
        *,
        store: CandidateStore,
        ticker_extractor: TickerExtractor,
        source_weights: Mapping[str, float] = DEFAULT_SOURCE_WEIGHTS,
        half_life_seconds: float = DEFAULT_HALF_LIFE_SECONDS,
        logger: logging.Logger | None = None,
        clock: "callable[[], float] | None" = None,
    ) -> None:
        self._sources = list(sources)
        self._store = store
        self._extractor = ticker_extractor
        self._weights = dict(source_weights)
        self._half_life = max(60.0, float(half_life_seconds))
        self._log = logger or logging.getLogger("etrader.news.aggregator")
        self._clock = clock or time.time

    # ------------------------------------------------------------------

    @property
    def sources(self) -> tuple[NewsSource, ...]:
        """Read-only view of the configured source plug-ins."""
        return tuple(self._sources)

    def source_names(self) -> tuple[str, ...]:
        """Return the canonical ``name`` of each wired source, in order."""
        return tuple(
            getattr(s, "name", s.__class__.__name__) for s in self._sources
        )

    def get_source(self, name: str) -> NewsSource | None:
        """Look up a wired source by its ``name`` attribute (case-insensitive)."""
        target = (name or "").strip().lower()
        if not target:
            return None
        for s in self._sources:
            if str(getattr(s, "name", s.__class__.__name__)).lower() == target:
                return s
        return None

    def source_weight(self, name: str) -> float:
        """Effective per-source weight multiplier (matches the live scoring)."""
        return float(self._weights.get(name, 0.5))

    def run(
        self,
        *,
        since: float | None = None,
        known_symbols: Iterable[str] | None = None,
    ) -> AggregatorRunStats:
        """Fan out across all sources and fold results into the store.

        Returns a small stats object so callers (the scheduler, a
        ``/news`` Telegram command, tests) can log / display the run.
        """
        stats = AggregatorRunStats(started_at_unix=self._clock())
        known_list = list(known_symbols) if known_symbols is not None else None

        seen_keys: set[str] = set()
        for source in self._sources:
            source_name = getattr(source, "name", source.__class__.__name__)
            try:
                items = list(source.fetch(since=since, known_symbols=known_list))
            except Exception as exc:  # noqa: BLE001 — never let one source kill the run
                self._log.warning("news source %s raised: %s", source_name, exc)
                stats.per_source_errors[source_name] = str(exc)
                continue

            stats.items_fetched += len(items)
            kept = 0
            for item in items:
                if not isinstance(item, NewsItem):
                    continue
                key = item.dedup_key
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                kept += 1
                self._fold_item(item, stats=stats)
            stats.items_kept += kept
            stats.per_source_counts[source_name] = kept

        stats.finished_at_unix = self._clock()
        try:
            self._store.save()
        except OSError as exc:
            self._log.warning("candidate store save failed: %s", exc)
        return stats

    # ------------------------------------------------------------------

    def _fold_item(self, item: NewsItem, *, stats: AggregatorRunStats) -> None:
        symbols = self._resolve_symbols(item)
        if not symbols:
            return
        base_weight = self._weights.get(item.source, 0.5)
        decay = self._freshness_decay(item.published_at)
        sentiment_boost = self._sentiment_boost(item.sentiment)
        weight = base_weight * decay * sentiment_boost
        if weight <= 0:
            return
        for symbol in symbols:
            self._store.record(
                symbol=symbol,
                source=item.source,
                headline=item.headline,
                weight=weight,
            )
            stats.observations_recorded += 1

    def _resolve_symbols(self, item: NewsItem) -> tuple[str, ...]:
        if item.symbols:
            return item.symbols
        # Run the extractor over headline + body. Combining gives both
        # the title-only sources (RSS summaries) and full-body sources
        # the same treatment.
        text = " ".join(filter(None, (item.headline, item.raw_text)))
        if not text:
            return ()
        extracted = self._extractor.extract(text)
        return extracted.symbols

    def _freshness_decay(self, published_at: float) -> float:
        """Exponential decay: weight ∝ 0.5 ** (age / half_life)."""
        if not published_at or published_at <= 0:
            return 1.0  # unknown publish-time → treat as "just now"
        age = max(0.0, self._clock() - float(published_at))
        return float(math.pow(0.5, age / self._half_life))

    @staticmethod
    def _sentiment_boost(sentiment: float | None) -> float:
        """Map sentiment ∈ [-1, +1] → multiplier ∈ [1.0, 1.5].

        We don't down-weight negative news (a -1 catalyst is just as
        tradable as a +1 catalyst). Magnitude is what matters here.
        """
        if sentiment is None:
            return 1.0
        magnitude = min(1.0, abs(float(sentiment)))
        return 1.0 + 0.5 * magnitude
