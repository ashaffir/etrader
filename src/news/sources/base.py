"""Common types for the news-source plug-in interface.

Every concrete source returns an iterable of :class:`NewsItem`. The
aggregator does not care how a source obtained its items (HTTP, RSS, an
SDK call, a local cache); it only cares about the shared fields.

Design notes
------------
* ``symbols`` is a tuple, not a list — items are treated as immutable and
  may be deduped across sources by ``(source, url)`` or by hashing.
* ``published_at`` is a Unix timestamp (float, seconds). UTC is implied;
  every source must normalise its native date format before emitting.
* ``sentiment`` is intentionally optional. Most free sources don't ship
  sentiment, and we'd rather store ``None`` than a fabricated number.
* ``raw_text`` holds the body / summary when available (used downstream
  for ticker extraction and LLM context); empty string when only a
  headline was provided.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class NewsItem:
    """A single news headline from any source.

    Attributes
    ----------
    source:
        Stable identifier of the originating source plug-in
        (``"stocktwits"``, ``"sec_edgar"``, ``"google_news"``, ``"yahoo_rss"``,
        ``"yfinance"``). Used for de-duplication and reason strings.
    symbols:
        Tickers explicitly associated with the item. Sources that emit
        ticker-tagged items (StockTwits, yfinance, SEC EDGAR after CIK
        resolution) populate this directly; query-based sources (Google
        News RSS) leave it empty and let the ticker extractor fill it in.
    headline:
        Short title text. Always non-empty.
    url:
        Canonical URL of the article. Used as a stable dedup key when
        combined with ``source``.
    published_at:
        Unix timestamp in seconds (UTC). ``0.0`` when the source did not
        provide a date and the item should be treated as "just now".
    raw_text:
        Optional body / summary text. Often used for richer ticker
        extraction and for LLM context. Empty if not provided.
    sentiment:
        Optional sentiment score in ``[-1.0, +1.0]`` when the source
        ships one (Finnhub, Alpha Vantage). ``None`` otherwise — never
        fabricate.
    metadata:
        Free-form, source-specific extras (e.g. StockTwits watcher count,
        SEC filing form type). Treated as opaque downstream.
    """

    source: str
    symbols: tuple[str, ...]
    headline: str
    url: str
    published_at: float = 0.0
    raw_text: str = ""
    sentiment: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def dedup_key(self) -> str:
        """Stable identifier used to drop cross-source duplicates."""
        return f"{self.source}::{self.url or self.headline}"


@runtime_checkable
class NewsSource(Protocol):
    """Plug-in contract for a news source.

    Implementations should be cheap to construct, fail soft (swallow
    transient errors and emit an empty iterable), and be safe to call
    repeatedly from the aggregator.
    """

    name: str

    def fetch(
        self,
        *,
        since: float | None = None,
        known_symbols: Iterable[str] | None = None,
    ) -> Iterable[NewsItem]:
        """Return zero or more :class:`NewsItem` objects.

        Parameters
        ----------
        since:
            Optional cutoff: ignore items older than this Unix timestamp.
            Sources that lack timestamps may ignore the hint.
        known_symbols:
            Optional iterable of tickers the bot already cares about.
            Per-ticker sources (yfinance, Yahoo per-ticker RSS) use this
            to scope their queries; broad/discovery sources ignore it.
        """
        ...
