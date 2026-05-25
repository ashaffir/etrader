"""StockTwits trending-symbols source.

StockTwits publishes a public, key-free endpoint that returns the
current top trending tickers, ranked by message volume across its
retail-trader social network. Each entry already comes with the ticker
pre-extracted, so we don't need to run the headline extractor on this
source — emit one :class:`NewsItem` per trending symbol.

Endpoint
--------
``GET https://api.stocktwits.com/api/2/trending/symbols.json``

Response shape (excerpt)::

    {
      "response": {"status": 200},
      "symbols": [
        {"id": 686, "symbol": "AAPL", "title": "Apple Inc.",
         "watchlist_count": 1234567, "instrument_class": "Stock", ...},
        ...
      ]
    }

We use ``watchlist_count`` as a soft popularity signal in the reason
string. The endpoint is documented at
https://api.stocktwits.com/developers/docs.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Iterable, Mapping

from .base import NewsItem


DEFAULT_URL = "https://api.stocktwits.com/api/2/trending/symbols.json"
DEFAULT_TIMEOUT = 10.0

# Sentinel — using a Protocol-shaped callable.
Fetcher = Callable[[str], Mapping[str, Any]]


def _default_fetcher(url: str) -> Mapping[str, Any]:
    """HTTP fetcher used when the caller doesn't inject one.

    Imported lazily so the package remains importable without
    ``requests`` (it ships in ``requirements.txt`` already, but lazy
    import keeps test surfaces clean).
    """
    import requests  # noqa: PLC0415 — lazy by design

    resp = requests.get(
        url,
        timeout=DEFAULT_TIMEOUT,
        headers={"User-Agent": "etrader/news (+stocktwits)", "Accept": "application/json"},
    )
    resp.raise_for_status()
    return resp.json()


class StockTwitsTrendingSource:
    """Emit one :class:`NewsItem` per ticker on the trending list.

    The headline is synthetic but human-readable
    (``"StockTwits trending: AAPL (Apple Inc.)"``), with the
    ``watchlist_count`` carried in metadata so the aggregator can weight
    accordingly.
    """

    name = "stocktwits"

    def __init__(
        self,
        *,
        url: str = DEFAULT_URL,
        fetcher: Fetcher | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._url = url
        self._fetcher = fetcher or _default_fetcher
        self._log = logger or logging.getLogger("etrader.news.stocktwits")

    def fetch(
        self,
        *,
        since: float | None = None,  # noqa: ARG002 — trending list is not time-indexed
        known_symbols: Iterable[str] | None = None,  # noqa: ARG002 — discovery source
    ) -> Iterable[NewsItem]:
        try:
            payload = self._fetcher(self._url)
        except Exception as exc:  # noqa: BLE001 — fail soft
            self._log.warning("stocktwits fetch failed: %s", exc)
            return []
        return list(_parse(payload))


def _parse(payload: Mapping[str, Any]) -> Iterable[NewsItem]:
    symbols = payload.get("symbols")
    if not isinstance(symbols, list):
        return
    now = time.time()
    for rank, entry in enumerate(symbols, start=1):
        if not isinstance(entry, Mapping):
            continue
        sym_raw = entry.get("symbol")
        if not isinstance(sym_raw, str) or not sym_raw.strip():
            continue
        symbol = sym_raw.strip().upper()
        title = str(entry.get("title") or "").strip()
        watchers_raw = entry.get("watchlist_count")
        try:
            watchers = int(watchers_raw) if watchers_raw is not None else None
        except (TypeError, ValueError):
            watchers = None
        instrument_class = str(entry.get("instrument_class") or "").strip()

        headline_bits = [f"StockTwits trending #{rank}: ${symbol}"]
        if title:
            headline_bits.append(f"({title})")
        if watchers is not None and watchers > 0:
            headline_bits.append(f"— {watchers:,} watchers")
        headline = " ".join(headline_bits)

        meta: dict[str, Any] = {"rank": rank}
        if watchers is not None:
            meta["watchlist_count"] = watchers
        if instrument_class:
            meta["instrument_class"] = instrument_class

        yield NewsItem(
            source="stocktwits",
            symbols=(symbol,),
            headline=headline,
            url=f"https://stocktwits.com/symbol/{symbol}",
            published_at=now,
            raw_text=title,
            sentiment=None,
            metadata=meta,
        )
