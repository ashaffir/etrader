"""Yahoo Finance per-ticker RSS source.

Yahoo's legacy RSS endpoint still ships per-ticker headline feeds:

    https://feeds.finance.yahoo.com/rss/2.0/headline?s=<TICKER>&region=US&lang=en-US

This is a complement to :class:`~src.news.sources.yfinance_news.YFinanceNewsSource`,
which scrapes the same data via the modern Yahoo Finance JSON endpoints.
Keeping both gives us a free fallback when one of them is rate-limited
or temporarily broken (Yahoo rearranges endpoints often). They are
deduplicated at aggregator level by URL.

The source is enrichment-only: it requires a ``known_symbols`` iterable
to know which tickers to query. Pass the empty list to skip cleanly.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable

from .base import NewsItem
from .google_news_rss import _entry_timestamp  # reuse — same feedparser shape


DEFAULT_BASE = "https://feeds.finance.yahoo.com/rss/2.0/headline"
DEFAULT_TIMEOUT = 10.0
DEFAULT_MAX_TICKERS = 50  # safety cap — avoids spraying when known list is huge

ParsedFeed = dict[str, Any]
Fetcher = Callable[[str], ParsedFeed]


def _default_fetcher(url: str) -> ParsedFeed:
    """Lazy-import feedparser + requests."""
    import feedparser  # noqa: PLC0415
    import requests  # noqa: PLC0415

    resp = requests.get(
        url,
        timeout=DEFAULT_TIMEOUT,
        headers={"User-Agent": "etrader/news (+yahoo_rss)"},
    )
    resp.raise_for_status()
    return feedparser.parse(resp.content)


def _build_url(base: str, symbol: str) -> str:
    return f"{base}?s={symbol}&region=US&lang=en-US"


class YahooRssSource:
    """Fetch the per-ticker Yahoo Finance RSS feed.

    Always sets ``symbols=(ticker,)`` on emitted items since the feed
    is already scoped to one symbol — no extraction step needed.
    """

    name = "yahoo_rss"

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE,
        max_symbols: int = DEFAULT_MAX_TICKERS,
        max_items_per_symbol: int = 10,
        fetcher: Fetcher | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._base = base_url
        self._max_symbols = max(1, int(max_symbols))
        self._max_per_symbol = max(1, int(max_items_per_symbol))
        self._fetcher = fetcher or _default_fetcher
        self._log = logger or logging.getLogger("etrader.news.yahoo_rss")

    def fetch(
        self,
        *,
        since: float | None = None,
        known_symbols: Iterable[str] | None = None,
    ) -> Iterable[NewsItem]:
        if known_symbols is None:
            return []
        # Preserve order, dedupe, cap.
        seen: set[str] = set()
        targets: list[str] = []
        for sym in known_symbols:
            if not isinstance(sym, str):
                continue
            s = sym.strip().upper()
            if not s or s in seen:
                continue
            seen.add(s)
            targets.append(s)
            if len(targets) >= self._max_symbols:
                break

        out: list[NewsItem] = []
        for symbol in targets:
            url = _build_url(self._base, symbol)
            try:
                feed = self._fetcher(url)
            except Exception as exc:  # noqa: BLE001 — fail soft per-ticker
                self._log.warning("yahoo_rss fetch failed (%s): %s", symbol, exc)
                continue
            out.extend(_parse_feed(feed, symbol=symbol, since=since, limit=self._max_per_symbol))
        return out


def _parse_feed(
    feed: ParsedFeed,
    *,
    symbol: str,
    since: float | None,
    limit: int,
) -> Iterable[NewsItem]:
    entries = feed.get("entries") or []
    if not isinstance(entries, list):
        return
    count = 0
    for entry in entries:
        if count >= limit:
            break
        title = str(entry.get("title") or "").strip()
        link = str(entry.get("link") or "").strip()
        if not title or not link:
            continue
        published = _entry_timestamp(entry)
        if since is not None and published and published < since:
            continue
        summary = str(entry.get("summary") or entry.get("description") or "").strip()
        yield NewsItem(
            source="yahoo_rss",
            symbols=(symbol,),
            headline=title,
            url=link,
            published_at=published,
            raw_text=summary,
            sentiment=None,
            metadata={"queried_symbol": symbol},
        )
        count += 1
