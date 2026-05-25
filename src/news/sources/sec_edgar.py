"""SEC EDGAR 8-K Atom-feed source.

8-K filings flag *material events* — earnings, M&A, restatements,
guidance changes, executive turnover, etc. The SEC publishes a free,
key-free Atom feed of the most recent 8-Ks:

    https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent
        &type=8-K&output=atom&count=40

Each entry's title looks like::

    8-K - APPLE INC (0000320193) (Filer)

To turn the CIK into a tradeable ticker we use the SEC's own mapping
file:

    https://www.sec.gov/files/company_tickers.json

This file is shipped daily by the SEC and is genuinely the canonical
CIK ↔ ticker map for US-listed issuers. We persist it under
``data/sec_cik_to_ticker.json`` and refresh on a weekly cadence.

The SEC requires a real ``User-Agent`` header on all programmatic
hits — see https://www.sec.gov/os/accessing-edgar-data. We make this
configurable via ``SEC_USER_AGENT`` (env var) with a sensible default.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .base import NewsItem


DEFAULT_FEED_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent"
    "&type=8-K&output=atom&count=40"
)
DEFAULT_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
DEFAULT_CACHE_PATH = Path("data") / "sec_cik_to_ticker.json"
DEFAULT_TIMEOUT = 15.0
TICKERS_REFRESH_SECONDS = 7 * 24 * 60 * 60  # 7 days

# Title format: "8-K - COMPANY NAME (CIK_DIGITS) (Filer)"
_TITLE_RE = re.compile(r"\((\d{4,10})\)\s*\(Filer\)", re.IGNORECASE)

ParsedFeed = dict[str, Any]
FeedFetcher = Callable[[str], ParsedFeed]
JsonFetcher = Callable[[str], dict[str, Any]]


_USER_AGENT_HINT = (
    "Set `SEC_USER_AGENT` in .env to a real contact string, e.g. "
    "`SEC_USER_AGENT=\"Jane Doe jane@example.com\"` — SEC blocks generic UAs."
)


def _resolve_user_agent() -> str | None:
    """Return the operator-configured SEC User-Agent, or ``None`` if missing.

    The SEC's Public-API access policy (see
    https://www.sec.gov/os/accessing-edgar-data) requires every
    automated client to identify itself with a name + contact email.
    Generic strings get a hard 403, so we treat "missing" as a real
    config error and skip the source until it's fixed.
    """
    explicit = os.environ.get("SEC_USER_AGENT")
    if explicit and "@" in explicit and len(explicit.strip()) >= 5:
        return explicit.strip()
    return None


def _default_feed_fetcher(url: str) -> ParsedFeed:
    """Lazy feedparser + requests fetch with SEC-compliant headers."""
    import feedparser  # noqa: PLC0415
    import requests  # noqa: PLC0415

    ua = _resolve_user_agent()
    if ua is None:
        raise RuntimeError(f"SEC_USER_AGENT not configured. {_USER_AGENT_HINT}")
    resp = requests.get(
        url,
        timeout=DEFAULT_TIMEOUT,
        headers={"User-Agent": ua, "Accept": "application/atom+xml"},
    )
    resp.raise_for_status()
    return feedparser.parse(resp.content)


def _default_tickers_fetcher(url: str) -> dict[str, Any]:
    """Lazy requests-based JSON fetch."""
    import requests  # noqa: PLC0415

    ua = _resolve_user_agent()
    if ua is None:
        raise RuntimeError(f"SEC_USER_AGENT not configured. {_USER_AGENT_HINT}")
    resp = requests.get(
        url,
        timeout=DEFAULT_TIMEOUT,
        headers={"User-Agent": ua, "Accept": "application/json"},
    )
    resp.raise_for_status()
    return resp.json()


@dataclass
class CikTickerMap:
    """CIK (int) → ticker mapping with disk persistence and TTL refresh.

    The SEC ships ``company_tickers.json`` as::

        {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}

    We flatten that to ``{cik: ticker}`` so lookups are O(1).
    """

    path: Path
    cik_to_ticker: dict[int, str]
    last_refresh_unix: float

    @classmethod
    def load(cls, path: Path) -> "CikTickerMap":
        if not path.exists():
            return cls(path=path, cik_to_ticker={}, last_refresh_unix=0.0)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return cls(path=path, cik_to_ticker={}, last_refresh_unix=0.0)
        mapping_raw = raw.get("cik_to_ticker") or {}
        mapping: dict[int, str] = {}
        for k, v in mapping_raw.items():
            try:
                mapping[int(k)] = str(v).upper()
            except (TypeError, ValueError):
                continue
        last_refresh = float(raw.get("last_refresh_unix") or 0.0)
        return cls(path=path, cik_to_ticker=mapping, last_refresh_unix=last_refresh)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = {
            "cik_to_ticker": {str(k): v for k, v in self.cik_to_ticker.items()},
            "last_refresh_unix": self.last_refresh_unix,
        }
        self.path.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")

    def needs_refresh(self, ttl_seconds: float = TICKERS_REFRESH_SECONDS) -> bool:
        if not self.cik_to_ticker:
            return True
        return (time.time() - self.last_refresh_unix) > ttl_seconds

    def update_from_payload(self, payload: dict[str, Any]) -> int:
        """Replace map contents from a fresh company_tickers.json blob.

        Returns the number of rows ingested. Robust against the
        occasional malformed entry — bad rows are skipped silently.
        """
        new_map: dict[int, str] = {}
        for _, row in payload.items():
            if not isinstance(row, dict):
                continue
            cik_raw = row.get("cik_str")
            ticker_raw = row.get("ticker")
            try:
                cik = int(cik_raw) if cik_raw is not None else None
            except (TypeError, ValueError):
                cik = None
            if cik is None or not isinstance(ticker_raw, str):
                continue
            ticker = ticker_raw.strip().upper()
            if not ticker:
                continue
            new_map[cik] = ticker
        self.cik_to_ticker = new_map
        self.last_refresh_unix = time.time()
        return len(new_map)

    def lookup(self, cik: int) -> str | None:
        return self.cik_to_ticker.get(int(cik))


class SecEdgar8KSource:
    """Stream 8-K filings as :class:`NewsItem` instances, ticker-resolved."""

    name = "sec_edgar"

    def __init__(
        self,
        *,
        feed_url: str = DEFAULT_FEED_URL,
        tickers_url: str = DEFAULT_TICKERS_URL,
        cache_path: Path = DEFAULT_CACHE_PATH,
        feed_fetcher: FeedFetcher | None = None,
        tickers_fetcher: JsonFetcher | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._feed_url = feed_url
        self._tickers_url = tickers_url
        self._cache_path = cache_path
        # Detect missing SEC_USER_AGENT at construction time so we can
        # skip the source entirely instead of getting 403'd every scan.
        # When the operator hasn't supplied custom fetchers (the normal
        # production path) we honor the env-var requirement; when tests
        # inject fetchers we trust them and skip the check.
        self._disabled_reason: str | None = None
        if feed_fetcher is None and tickers_fetcher is None and _resolve_user_agent() is None:
            self._disabled_reason = (
                "SEC_USER_AGENT not set or missing '@' — source disabled. "
                + _USER_AGENT_HINT
            )
        self._feed_fetcher = feed_fetcher or _default_feed_fetcher
        self._tickers_fetcher = tickers_fetcher or _default_tickers_fetcher
        self._log = logger or logging.getLogger("etrader.news.sec_edgar")
        if self._disabled_reason:
            self._log.warning("[sec_edgar] %s", self._disabled_reason)
        self._lock = threading.RLock()
        self._cik_map = CikTickerMap.load(cache_path)

    def fetch(
        self,
        *,
        since: float | None = None,
        known_symbols: Iterable[str] | None = None,  # noqa: ARG002 — discovery source
    ) -> Iterable[NewsItem]:
        if self._disabled_reason:
            # Quiet on the hot path — the construction-time warning is
            # the operator-visible signal. Returning [] keeps the
            # aggregator happy.
            return []
        self._maybe_refresh_cik_map()
        try:
            feed = self._feed_fetcher(self._feed_url)
        except Exception as exc:  # noqa: BLE001 — fail soft
            self._log.warning("sec_edgar feed fetch failed: %s", exc)
            return []
        return list(self._parse_feed(feed, since=since))

    def _maybe_refresh_cik_map(self) -> None:
        with self._lock:
            if not self._cik_map.needs_refresh():
                return
        try:
            payload = self._tickers_fetcher(self._tickers_url)
        except Exception as exc:  # noqa: BLE001 — log and proceed with stale map
            self._log.warning("sec_edgar ticker map refresh failed: %s", exc)
            return
        with self._lock:
            count = self._cik_map.update_from_payload(payload)
            try:
                self._cik_map.save()
            except OSError as exc:
                self._log.warning("sec_edgar ticker map save failed: %s", exc)
            self._log.info("sec_edgar refreshed CIK→ticker map: %d entries", count)

    def _parse_feed(self, feed: ParsedFeed, *, since: float | None) -> Iterable[NewsItem]:
        from .google_news_rss import _entry_timestamp  # local — same shape

        entries = feed.get("entries") or []
        if not isinstance(entries, list):
            return
        for entry in entries:
            title = str(entry.get("title") or "").strip()
            link = str(entry.get("link") or "").strip()
            if not title or not link:
                continue
            published = _entry_timestamp(entry)
            if since is not None and published and published < since:
                continue
            cik = _cik_from_title(title)
            symbols: tuple[str, ...] = ()
            if cik is not None:
                ticker = self._cik_map.lookup(cik)
                if ticker:
                    symbols = (ticker,)
            summary = str(entry.get("summary") or entry.get("description") or "").strip()
            meta: dict[str, Any] = {"form_type": "8-K"}
            if cik is not None:
                meta["cik"] = cik
            yield NewsItem(
                source="sec_edgar",
                symbols=symbols,
                headline=title,
                url=link,
                published_at=published,
                raw_text=summary,
                sentiment=None,
                metadata=meta,
            )


def _cik_from_title(title: str) -> int | None:
    """Extract the CIK number from an EDGAR Atom entry title.

    EDGAR titles follow ``"8-K - <COMPANY> (<CIK>) (Filer)"``. We trust
    the ``(Filer)`` suffix so we don't accidentally pick up the CIK of
    a co-filer or reporter mentioned later in the title.
    """
    m = _TITLE_RE.search(title)
    if not m:
        return None
    try:
        return int(m.group(1))
    except (TypeError, ValueError):
        return None
