"""Market-data wrappers: search, instruments, rates, candles.

Each function returns plain dicts/dataclasses that the rest of the app
can consume without knowing about HTTP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .client import EtoroClient
from .errors import EtoroPayloadTooLargeError


# ---------------------------------------------------------------------------
# Live rates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LiveRate:
    instrument_id: int
    ask: float | None
    bid: float | None
    last: float | None
    timestamp: str | None

    @property
    def mid(self) -> float | None:
        if self.ask is None or self.bid is None:
            return self.last
        return (self.ask + self.bid) / 2.0


def _rate_from(raw: dict[str, Any]) -> LiveRate:
    return LiveRate(
        instrument_id=int(raw.get("instrumentID") or raw.get("instrumentId") or 0),
        ask=_maybe_float(raw.get("ask")),
        bid=_maybe_float(raw.get("bid")),
        last=_maybe_float(raw.get("lastExecution")),
        timestamp=raw.get("date"),
    )


def fetch_rates(client: EtoroClient, instrument_ids: Sequence[int]) -> dict[int, LiveRate]:
    """Live bid/ask/last for up to 100 instruments."""
    out: dict[int, LiveRate] = {}
    if not instrument_ids:
        return out
    # /market-data/instruments/rates accepts up to 100 IDs per call.
    chunk_size = 100
    for i in range(0, len(instrument_ids), chunk_size):
        chunk = list(instrument_ids[i : i + chunk_size])
        payload = client.get(
            "/market-data/instruments/rates",
            params={"instrumentIds": chunk},
            retries=2,
        )
        for raw in (payload or {}).get("rates", []) or []:
            rate = _rate_from(raw)
            if rate.instrument_id:
                out[rate.instrument_id] = rate
    return out


# ---------------------------------------------------------------------------
# Instrument metadata
# ---------------------------------------------------------------------------

# eToro's ``priceSource`` is the most reliable asset-class signal we
# get from /market-data/instruments:
#   - real equity exchanges (NASDAQ, NYSE, LSE, ...) → STOCK or ETF
#   - "eToro" → in-house crypto book
#   - currency tickers (USD, EUR, JPY, ...) → FX
# We use it in :func:`InstrumentMeta.is_crypto` and in
# :func:`src.strategy.tools.base.asset_class_for` to override the
# notoriously wrong ``instrumentTypeID`` field (eToro returns
# ``instrumentTypeID=5`` for cash equities like AMD/AAPL/LUNR and
# ``instrumentTypeID=10`` for crypto, which is the opposite of what
# the docs imply).
_ETORO_CRYPTO_PRICE_SOURCE = "etoro"


@dataclass(frozen=True)
class InstrumentMeta:
    instrument_id: int
    display_name: str | None
    symbol_full: str | None
    instrument_type_id: int | None
    exchange_id: int | None
    raw: dict[str, Any]

    @property
    def price_source(self) -> str | None:
        return self.raw.get("priceSource") if isinstance(self.raw, dict) else None

    @property
    def stocks_industry_id(self) -> int | None:
        if not isinstance(self.raw, dict):
            return None
        v = self.raw.get("stocksIndustryID") or self.raw.get("stocksIndustryId")
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    @property
    def is_crypto(self) -> bool:
        # Authoritative signal: eToro hosts every crypto in its in-house
        # book and stamps ``priceSource = "eToro"`` on them. Cash equities
        # carry a real exchange name (NASDAQ, NYSE, …). We deliberately
        # do NOT trust ``instrument_type_id`` here because eToro returns
        # 5 for stocks and 10 for crypto (verified live across 11
        # instruments) — the opposite of what one would guess.
        src = (self.price_source or "").strip().lower()
        if src == _ETORO_CRYPTO_PRICE_SOURCE:
            return True
        # Symbol fallback for stale snapshots / search payloads that
        # don't include priceSource. Keep this list narrow so we don't
        # accidentally mis-tag a stock like ADAP (Adaptimmune) as ADA.
        sym = (self.symbol_full or "").upper()
        return sym in {"BTC", "ETH", "SOL", "DOGE", "XRP", "LTC", "BCH"}


def _instrument_from(raw: dict[str, Any]) -> InstrumentMeta:
    return InstrumentMeta(
        instrument_id=int(raw.get("instrumentID") or raw.get("instrumentId") or 0),
        display_name=raw.get("instrumentDisplayName"),
        symbol_full=raw.get("symbolFull"),
        instrument_type_id=raw.get("instrumentTypeID") or raw.get("instrumentTypeId"),
        exchange_id=raw.get("exchangeID") or raw.get("exchangeId"),
        raw=raw,
    )


_INSTRUMENT_BATCH_LADDER = (50, 25, 10)


def fetch_instruments(
    client: EtoroClient,
    instrument_ids: Sequence[int],
) -> dict[int, InstrumentMeta]:
    """Resolve display-data for instrument IDs, with adaptive 413/414 batching."""
    out: dict[int, InstrumentMeta] = {}
    if not instrument_ids:
        return out

    pending = list(instrument_ids)
    batch_size = _INSTRUMENT_BATCH_LADDER[0]
    i = 0
    while i < len(pending):
        chunk = pending[i : i + batch_size]
        try:
            payload = client.get(
                "/market-data/instruments",
                params={"instrumentIds": chunk},
                retries=2,
            )
        except EtoroPayloadTooLargeError:
            try:
                idx = _INSTRUMENT_BATCH_LADDER.index(batch_size)
            except ValueError:
                idx = 0
            if idx + 1 >= len(_INSTRUMENT_BATCH_LADDER):
                # Already minimal; skip this chunk so we don't spin forever.
                i += batch_size
                continue
            batch_size = _INSTRUMENT_BATCH_LADDER[idx + 1]
            continue
        for raw in (payload or {}).get("instrumentDisplayDatas", []) or []:
            meta = _instrument_from(raw)
            if meta.instrument_id:
                out[meta.instrument_id] = meta
        i += batch_size
    return out


# ---------------------------------------------------------------------------
# Search (symbol → ID)
# ---------------------------------------------------------------------------

def search_instrument(client: EtoroClient, symbol: str) -> int | None:
    """Live-resolve a symbol to an instrument ID. Returns ``None`` if no match."""
    payload = client.get(
        "/market-data/search",
        params={"internalSymbolFull": symbol},
        retries=2,
    )
    items = (payload or {}).get("items") or []
    if not items:
        return None
    target = symbol.upper()
    for item in items:
        if str(item.get("symbolFull", "")).upper() == target:
            return _safe_int(item.get("instrumentId") or item.get("instrumentID"))
    return _safe_int(items[0].get("instrumentId") or items[0].get("instrumentID"))


# ---------------------------------------------------------------------------
# Candles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Candle:
    instrument_id: int
    from_date: str | None
    open: float
    high: float
    low: float
    close: float
    volume: float


def fetch_candles(
    client: EtoroClient,
    instrument_id: int,
    *,
    interval: str = "OneHour",
    count: int = 100,
    direction: str = "asc",
) -> list[Candle]:
    """Historical OHLCV candles for one instrument (asc → oldest first).

    eToro can return ``null`` for any of ``open / high / low / close`` —
    typically when an instrument was inactive during a slot (weekend,
    pre-listing, or thin crypto periods). Such candles are skipped so
    indicators never see junk values; ``volume`` is allowed to be null
    (treated as 0) since technical signals don't rely on it.
    """
    count = max(1, min(int(count), 1000))
    path = (
        f"/market-data/instruments/{instrument_id}/history/candles/"
        f"{direction}/{interval}/{count}"
    )
    payload = client.get(path, retries=2)
    candles: list[Candle] = []
    for entry in (payload or {}).get("candles", []) or []:
        for c in entry.get("candles", []) or []:
            o = _maybe_float(c.get("open"))
            h = _maybe_float(c.get("high"))
            low = _maybe_float(c.get("low"))
            close = _maybe_float(c.get("close"))
            if o is None or h is None or low is None or close is None:
                continue  # skip degenerate candle (eToro returned null OHLCV)
            if close <= 0:
                continue
            candles.append(
                Candle(
                    instrument_id=int(
                        c.get("instrumentID") or c.get("instrumentId") or instrument_id
                    ),
                    from_date=c.get("fromDate"),
                    open=o,
                    high=h,
                    low=low,
                    close=close,
                    volume=_maybe_float(c.get("volume")) or 0.0,
                )
            )
    return candles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _maybe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
