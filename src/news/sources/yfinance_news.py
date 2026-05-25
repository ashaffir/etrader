"""yfinance ``Ticker.news`` source.

For each known symbol, yfinance returns a list of headlines pulled
from Yahoo Finance's JSON endpoints. The shape (as of yfinance 0.2.x)
is a list of dicts with the keys::

    title, link, publisher, providerPublishTime, type, relatedTickers

``relatedTickers`` is gold for discovery — co-mentioned symbols often
identify the next candidate before any aggregator processing. We carry
them through ``NewsItem.symbols`` so the aggregator can promote them.

yfinance is imported lazily (via a helper) so the rest of the news
pipeline remains importable and testable on machines where yfinance
isn't installed yet.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Sequence

from .base import NewsItem


DEFAULT_MAX_TICKERS = 50

NewsFetcher = Callable[[str], Sequence[dict[str, Any]]]


def _default_news_fetcher(symbol: str) -> Sequence[dict[str, Any]]:
    """Default fetcher — hits ``yfinance.Ticker(symbol).news``.

    Lazy-imports yfinance so importing this module never raises.
    """
    import yfinance as yf  # noqa: PLC0415 — lazy by design

    ticker = yf.Ticker(symbol)
    raw = getattr(ticker, "news", None) or []
    return [r for r in raw if isinstance(r, dict)]


class YFinanceNewsSource:
    """Fetch per-symbol yfinance news, surfacing ``relatedTickers``.

    Items are emitted with ``symbols=(queried_symbol, *related)`` so
    the aggregator sees the primary symbol plus any related tickers
    yfinance suggested.
    """

    name = "yfinance"

    def __init__(
        self,
        *,
        max_symbols: int = DEFAULT_MAX_TICKERS,
        max_items_per_symbol: int = 10,
        fetcher: NewsFetcher | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._max_symbols = max(1, int(max_symbols))
        self._max_per_symbol = max(1, int(max_items_per_symbol))
        self._fetcher = fetcher or _default_news_fetcher
        self._log = logger or logging.getLogger("etrader.news.yfinance")

    def fetch(
        self,
        *,
        since: float | None = None,
        known_symbols: Iterable[str] | None = None,
    ) -> Iterable[NewsItem]:
        if known_symbols is None:
            return []
        targets = _normalise_targets(known_symbols, self._max_symbols)
        out: list[NewsItem] = []
        for symbol in targets:
            try:
                items = self._fetcher(symbol)
            except Exception as exc:  # noqa: BLE001 — fail soft per-ticker
                self._log.warning("yfinance news fetch failed (%s): %s", symbol, exc)
                continue
            out.extend(_parse_items(items, queried=symbol, since=since, limit=self._max_per_symbol))
        return out


def _normalise_targets(known: Iterable[str], cap: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for sym in known:
        if not isinstance(sym, str):
            continue
        s = sym.strip().upper()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= cap:
            break
    return out


def _parse_items(
    items: Sequence[dict[str, Any]],
    *,
    queried: str,
    since: float | None,
    limit: int,
) -> Iterable[NewsItem]:
    count = 0
    for raw in items:
        if count >= limit:
            break
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title") or "").strip()
        link = str(raw.get("link") or raw.get("url") or "").strip()
        if not title or not link:
            continue
        try:
            published = float(raw.get("providerPublishTime") or 0.0)
        except (TypeError, ValueError):
            published = 0.0
        if since is not None and published and published < since:
            continue
        related = raw.get("relatedTickers") or []
        related_symbols: list[str] = [queried]
        if isinstance(related, list):
            for r in related:
                if isinstance(r, str):
                    s = r.strip().upper()
                    if s and s not in related_symbols:
                        related_symbols.append(s)
        publisher = str(raw.get("publisher") or "").strip()
        meta: dict[str, Any] = {"queried_symbol": queried}
        if publisher:
            meta["publisher"] = publisher
        if "type" in raw and raw["type"]:
            meta["type"] = str(raw["type"]).strip()
        yield NewsItem(
            source="yfinance",
            symbols=tuple(related_symbols),
            headline=title,
            url=link,
            published_at=published,
            raw_text="",
            sentiment=None,
            metadata=meta,
        )
        count += 1
