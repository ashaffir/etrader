"""yfinance-backed earnings calendar with persistent TTL cache.

Free data, no API key, no rate limits beyond what Yahoo enforces on
unauthenticated traffic. We use it to answer one question per symbol:

    "When is the next scheduled earnings call?"

The bot then surfaces ``days_to_earnings`` / ``hours_to_earnings`` into
the LLM payload and the strategy rule layer enforces:

* a guardrail (``pre_earnings_buy_blackout_hours``) that refuses BUYs
  for any name with earnings inside that window — earnings risk is a
  fundamental gamble the technical signal can't predict;
* a directive (``pre_earnings_close_hours``) that flattens any open
  bot position when its underlying is inside the window — the same
  rationale but applied to existing exposure.

Design constraints:

* Non-fatal. yfinance calls are wrapped; a failure logs and yields
  ``None``. Trading must not be blocked by Yahoo flakiness.
* Cached. We persist results to ``data/earnings_cache.json`` with a
  configurable TTL (default 12h) so a restart doesn't trigger N
  Yahoo round-trips.
* Future-only. Past earnings dates are dropped; only the next upcoming
  one is exposed. The technical pipeline doesn't need history here.
* Optional. Disabled (`enabled=False`) until the operator opts in,
  same pattern as the cross-asset regime cache.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


DEFAULT_TTL_SECONDS = 12 * 3600  # 12h: earnings dates don't move intraday
_FUTURE_HORIZON_DAYS = 120        # ignore anything more than 4 months out


@dataclass(frozen=True)
class EarningsEntry:
    """One scheduled earnings event for a symbol."""

    symbol: str
    earnings_at_utc: datetime
    fetched_at_unix: float
    source: str = "yfinance"

    def hours_until(self, now: datetime | None = None) -> float:
        now = now or datetime.now(timezone.utc)
        delta = self.earnings_at_utc - now
        return delta.total_seconds() / 3600.0

    def days_until(self, now: datetime | None = None) -> int:
        return int(self.hours_until(now) // 24)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "earnings_at_utc": self.earnings_at_utc.isoformat(),
            "fetched_at_unix": self.fetched_at_unix,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EarningsEntry | None":
        try:
            iso = str(payload["earnings_at_utc"])
            when = datetime.fromisoformat(iso)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            return cls(
                symbol=str(payload["symbol"]).upper(),
                earnings_at_utc=when,
                fetched_at_unix=float(payload.get("fetched_at_unix") or 0.0),
                source=str(payload.get("source") or "yfinance"),
            )
        except (KeyError, ValueError, TypeError):
            return None


# A "fetcher" is anything ``(symbol) -> EarningsEntry | None``. The
# default uses yfinance; tests inject a fake so they don't hit the
# network. The shape is a callable rather than a class so each
# concrete provider stays trivially testable.
Fetcher = Callable[[str], "EarningsEntry | None"]


def _default_fetcher(symbol: str) -> EarningsEntry | None:
    """Resolve the next earnings date for ``symbol`` via yfinance.

    yfinance changed its earnings API a few times; we try the newest
    ``Ticker.earnings_dates`` DataFrame first and fall back to
    ``Ticker.calendar`` when that's missing. Both paths convert to
    UTC and pick the soonest *future* timestamp.
    """
    try:
        import yfinance as yf  # noqa: PLC0415
    except ImportError:  # pragma: no cover — yfinance is in requirements
        return None

    try:
        ticker = yf.Ticker(symbol)
        when = _next_from_earnings_dates(ticker)
        if when is None:
            when = _next_from_calendar(ticker)
        if when is None:
            return None
        return EarningsEntry(
            symbol=symbol.upper(),
            earnings_at_utc=when,
            fetched_at_unix=datetime.now(timezone.utc).timestamp(),
        )
    except Exception:  # noqa: BLE001 — yfinance throws a wide variety
        return None


def _next_from_earnings_dates(ticker: Any) -> datetime | None:
    """Prefer ``earnings_dates`` (DataFrame indexed by datetime)."""
    try:
        df = ticker.earnings_dates
    except Exception:  # noqa: BLE001
        return None
    if df is None or getattr(df, "empty", True):
        return None
    now = datetime.now(timezone.utc)
    horizon = now + timedelta(days=_FUTURE_HORIZON_DAYS)
    best: datetime | None = None
    for raw_idx in df.index:
        when = _coerce_utc(raw_idx)
        if when is None:
            continue
        if when < now or when > horizon:
            continue
        if best is None or when < best:
            best = when
    return best


def _next_from_calendar(ticker: Any) -> datetime | None:
    """Fall back to ``Ticker.calendar`` (dict-shaped on newer yfinance)."""
    try:
        cal = ticker.calendar
    except Exception:  # noqa: BLE001
        return None
    if not cal:
        return None
    candidates: list[Any] = []
    if isinstance(cal, dict):
        candidates = list(cal.get("Earnings Date") or [])
    else:
        try:
            candidates = list(cal.loc["Earnings Date"].dropna().tolist())  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            candidates = []
    now = datetime.now(timezone.utc)
    best: datetime | None = None
    for raw in candidates:
        when = _coerce_utc(raw)
        if when is None or when < now:
            continue
        if best is None or when < best:
            best = when
    return best


def _coerce_utc(value: Any) -> datetime | None:
    """Best-effort convert a yfinance datetime-ish to a tz-aware UTC dt."""
    if value is None:
        return None
    if hasattr(value, "to_pydatetime"):
        try:
            value = value.to_pydatetime()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


class EarningsCalendarCache:
    """Persistent, thread-safe TTL cache for next-earnings lookups.

    Designed to be queried once per cycle per symbol; the lookup is a
    pure dict read. The cache refreshes lazily — calling
    :meth:`refresh` for a symbol whose entry is fresh is a no-op. The
    cycle's universe builder runs a periodic batch :meth:`refresh_many`
    so the lookup path on the hot trade-decision codepath never blocks
    on Yahoo.
    """

    def __init__(
        self,
        path: Path,
        *,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        fetcher: Fetcher | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._path = Path(path)
        self._ttl = max(60, int(ttl_seconds))
        self._fetcher = fetcher or _default_fetcher
        self._log = logger or logging.getLogger("etrader.strategy.earnings")
        self._lock = threading.RLock()
        self._entries: dict[str, EarningsEntry] = {}
        self._negatives: dict[str, float] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, symbol: str) -> EarningsEntry | None:
        sym = symbol.upper().strip()
        if not sym:
            return None
        with self._lock:
            entry = self._entries.get(sym)
        if entry is None:
            return None
        # Drop entries whose earnings date is in the past (the next
        # call to refresh() will repopulate).
        if entry.earnings_at_utc < datetime.now(timezone.utc):
            with self._lock:
                self._entries.pop(sym, None)
            return None
        return entry

    def refresh(self, symbol: str, *, force: bool = False) -> EarningsEntry | None:
        """Fetch from yfinance if the cached entry is stale or absent.

        Returns the resulting entry (possibly ``None`` if Yahoo doesn't
        cover the symbol or the fetch failed). Negative results are
        cached for half a TTL window so we don't hammer Yahoo for
        delisted / unknown tickers.
        """
        sym = symbol.upper().strip()
        if not sym:
            return None
        now_unix = datetime.now(timezone.utc).timestamp()
        with self._lock:
            existing = self._entries.get(sym)
            if not force and existing is not None:
                age = now_unix - existing.fetched_at_unix
                if age < self._ttl:
                    return existing
            negative_at = self._negatives.get(sym)
            if not force and negative_at is not None:
                if now_unix - negative_at < (self._ttl / 2):
                    return None
        try:
            entry = self._fetcher(sym)
        except Exception as exc:  # noqa: BLE001 — fetcher must never throw
            self._log.warning("[earnings] fetch failed for %s: %s", sym, exc)
            entry = None
        with self._lock:
            if entry is None:
                self._negatives[sym] = now_unix
                self._entries.pop(sym, None)
            else:
                self._entries[sym] = entry
                self._negatives.pop(sym, None)
            self._save()
        return entry

    def refresh_many(self, symbols: Iterable[str], *, force: bool = False) -> None:
        for sym in symbols:
            self.refresh(sym, force=force)

    def snapshot(self) -> dict[str, EarningsEntry]:
        with self._lock:
            return dict(self._entries)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self._log.warning("[earnings] cannot read %s: %s", self._path, exc)
            return
        entries = raw.get("entries") or {}
        if not isinstance(entries, dict):
            return
        loaded = 0
        for sym, payload in entries.items():
            if not isinstance(payload, dict):
                continue
            entry = EarningsEntry.from_dict(payload)
            if entry is None:
                continue
            self._entries[sym.upper()] = entry
            loaded += 1
        negatives = raw.get("negatives") or {}
        if isinstance(negatives, dict):
            for sym, ts in negatives.items():
                try:
                    self._negatives[sym.upper()] = float(ts)
                except (TypeError, ValueError):
                    continue
        if loaded:
            self._log.debug("[earnings] loaded %d cached entries", loaded)

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "entries": {
                    sym: e.to_dict() for sym, e in self._entries.items()
                },
                "negatives": dict(self._negatives),
            }
            self._path.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except OSError as exc:
            self._log.warning("[earnings] cannot persist %s: %s", self._path, exc)
