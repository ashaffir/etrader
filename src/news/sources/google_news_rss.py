"""Google News RSS source — broad, query-based discovery.

Google News exposes an RSS feed for any search query, with no API key
and no rate limit in practice for the modest volumes we need:

    https://news.google.com/rss/search?q=<encoded>&hl=en-US&gl=US&ceid=US:en

This source rotates through a configurable list of finance-oriented
queries and aggregates headlines from all of them. The aggregator's
ticker extractor is responsible for turning the prose into symbol
candidates — we don't try to do extraction here.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence
from urllib.parse import quote_plus

from .base import NewsItem


DEFAULT_QUERIES: tuple[str, ...] = (
    "stock market news today",
    "earnings beat",
    "earnings miss",
    "stock upgrade analyst",
    "stock downgrade",
    "merger acquisition deal",
    "FDA approval drug",
    "guidance raised",
    "stock crash plunge",
    "stock surge soar",
    "crypto bitcoin ethereum news",
)
DEFAULT_BASE = "https://news.google.com/rss/search"
DEFAULT_TIMEOUT = 10.0


@dataclass(frozen=True)
class GoogleNewsLocale:
    """A ``(hl, gl, ceid)`` triple selecting a Google News country edition.

    Google News routes queries through country editions, so swapping
    these three fields lets us pull the same finance queries from
    London, Frankfurt, Tokyo, Hong Kong, Sydney, etc. The ``label``
    field is what gets stamped into :class:`NewsItem.metadata` for
    debugging / telemetry.
    """

    label: str   # "US" / "UK" / "DE" / "JP" / "HK" / "AU" …
    hl: str      # interface language ("en-US", "en-GB", "de", "ja", …)
    gl: str      # geo (country code: "US", "GB", "DE", …)
    ceid: str    # country edition id ("US:en", "GB:en", "DE:de", …)


US_LOCALE = GoogleNewsLocale(label="US", hl="en-US", gl="US", ceid="US:en")

# A starter set of finance-relevant locales. Editing this tuple is
# the cheap way to add coverage; each entry below produces one
# feed-per-query when used.
REGIONAL_LOCALES: tuple[GoogleNewsLocale, ...] = (
    GoogleNewsLocale(label="UK", hl="en-GB", gl="GB", ceid="GB:en"),
    GoogleNewsLocale(label="DE", hl="de",    gl="DE", ceid="DE:de"),
    GoogleNewsLocale(label="FR", hl="fr",    gl="FR", ceid="FR:fr"),
    GoogleNewsLocale(label="JP", hl="ja",    gl="JP", ceid="JP:ja"),
    GoogleNewsLocale(label="HK", hl="en-HK", gl="HK", ceid="HK:en"),
    GoogleNewsLocale(label="AU", hl="en-AU", gl="AU", ceid="AU:en"),
)


REGIONAL_QUERIES: tuple[str, ...] = (
    # Index-name probes — Google News attaches lots of constituent
    # stories to these terms, so the ticker extractor (vocab seeded
    # from base_symbols) finds plenty of mentions per region.
    "FTSE 100",
    "DAX 40",
    "CAC 40",
    "Nikkei 225",
    "Hang Seng",
    "ASX 200",
    # Generic — phrased so it works in non-English editions too.
    "stock market today",
    "earnings",
)

# Each fetcher returns the *parsed* feed dict (feedparser-style) so the
# real and fake fetchers share a single interface.
ParsedFeed = dict[str, Any]
Fetcher = Callable[[str], ParsedFeed]


def _default_fetcher(url: str) -> ParsedFeed:
    """Lazy-import feedparser + requests and return a parsed feed.

    Using ``requests`` for the actual fetch (so we control timeout +
    user agent) then handing the raw body to ``feedparser`` decouples
    HTTP from RSS parsing. Easier to mock, kinder to feedparser's
    sometimes-flaky network code.
    """
    import feedparser  # noqa: PLC0415
    import requests  # noqa: PLC0415

    resp = requests.get(
        url,
        timeout=DEFAULT_TIMEOUT,
        headers={"User-Agent": "etrader/news (+google_news_rss)"},
    )
    resp.raise_for_status()
    return feedparser.parse(resp.content)


def _build_url(base: str, query: str, locale: GoogleNewsLocale) -> str:
    return (
        f"{base}?q={quote_plus(query)}"
        f"&hl={locale.hl}&gl={locale.gl}&ceid={locale.ceid}"
    )


class GoogleNewsRssSource:
    """Aggregate Google News RSS entries across a list of finance queries.

    ``locale`` selects the country edition. Defaults to the US edition
    so existing config files keep working unchanged. Pass any
    :class:`GoogleNewsLocale` to pull foreign headlines instead — see
    :data:`REGIONAL_LOCALES`.

    ``source_name_suffix`` is appended to the ``source`` field of every
    emitted :class:`NewsItem` so downstream stats can distinguish
    ``google_news_us`` from ``google_news_uk`` even though they share
    the same class.
    """

    name = "google_news"

    def __init__(
        self,
        *,
        queries: Sequence[str] = DEFAULT_QUERIES,
        locale: GoogleNewsLocale = US_LOCALE,
        base_url: str = DEFAULT_BASE,
        max_items_per_query: int = 25,
        fetcher: Fetcher | None = None,
        source_name_suffix: str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._queries = tuple(q for q in queries if q and q.strip())
        self._locale = locale
        self._base = base_url
        self._max_per_query = max(1, int(max_items_per_query))
        self._fetcher = fetcher or _default_fetcher
        suffix = (source_name_suffix or locale.label or "").lower()
        self._source_label = f"google_news_{suffix}" if suffix else "google_news"
        self._log = logger or logging.getLogger("etrader.news.google_rss")

    def fetch(
        self,
        *,
        since: float | None = None,
        known_symbols: Iterable[str] | None = None,  # noqa: ARG002 — broad source
    ) -> Iterable[NewsItem]:
        out: list[NewsItem] = []
        for query in self._queries:
            url = _build_url(self._base, query, self._locale)
            try:
                feed = self._fetcher(url)
            except Exception as exc:  # noqa: BLE001 — fail soft per source
                self._log.warning(
                    "%s fetch failed (%s): %s", self._source_label, query, exc,
                )
                continue
            out.extend(_parse_feed(
                feed,
                query=query,
                locale=self._locale.label,
                source_label=self._source_label,
                since=since,
                limit=self._max_per_query,
            ))
        return out


def _parse_feed(
    feed: ParsedFeed,
    *,
    query: str,
    locale: str,
    source_label: str,
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
            source=source_label,
            symbols=(),
            headline=title,
            url=link,
            published_at=published,
            raw_text=summary,
            sentiment=None,
            metadata={"query": query, "locale": locale},
        )
        count += 1


def _entry_timestamp(entry: dict[str, Any]) -> float:
    """Best-effort published-at extraction from a feedparser entry.

    feedparser populates either ``published_parsed`` or ``updated_parsed``
    as a :class:`time.struct_time`. Fall back to ``0.0`` (treated as
    "unknown / just now") rather than guessing.
    """
    for key in ("published_parsed", "updated_parsed"):
        st = entry.get(key)
        if st is not None:
            try:
                # feedparser returns UTC struct_time; calendar.timegm is the
                # right inverse, but stdlib's time.mktime is local. Use
                # calendar to avoid TZ skew.
                import calendar  # noqa: PLC0415

                return float(calendar.timegm(st))
            except (TypeError, ValueError, OverflowError):
                continue
    return 0.0
