"""Domain types for the fundamentals subsystem.

The fundamentals layer mirrors the structure of the news pipeline:

- :class:`FundamentalsSnapshot` is the plain-data record stored in the
  on-disk cache. It is what every consumer (universe builder, LLM
  prompt, Telegram ``/fundamentals``) sees.
- :class:`FundamentalsFetcher` is a tiny Protocol so the cache can be
  unit-tested without yfinance / network access; the production
  implementation lives in :mod:`.yfinance_fetcher`.

All numeric fields are kept ``float | None`` so a single missing /
malformed value never breaks downstream rendering. The cache is
deliberately conservative about what it persists — only the well-known
fields the LLM and the operator UI actually use.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol


# Every field on the snapshot is optional: yfinance returns wildly
# different shapes across asset classes (stocks vs ETFs vs crypto vs FX)
# and we don't want a missing P/E ratio to mean "no fundamentals at all
# for this ticker".

@dataclass
class FundamentalsSnapshot:
    """Per-symbol fundamentals frozen at a point in time.

    The struct is shallow and JSON-stable so it round-trips cleanly
    through ``data/fundamentals_cache.json`` and the HTTP control API.

    ``fetched_at_unix`` is the wallclock when the data was scraped;
    callers use it to decide whether to refresh (see
    :class:`FundamentalsConfig.refresh_after_hours`).

    ``next_earnings_unix`` is also a freshness signal: when the
    earnings timestamp passes, we should re-fetch — quarterly results
    typically reset the fundamentals picture in a single trading day.
    """

    symbol: str
    fetched_at_unix: float

    # Identity / classification
    name: str | None = None
    exchange: str | None = None
    quote_type: str | None = None  # EQUITY | ETF | CRYPTOCURRENCY | INDEX | ...
    sector: str | None = None
    industry: str | None = None
    country: str | None = None
    currency: str | None = None

    # Headline valuation
    market_cap: float | None = None
    enterprise_value: float | None = None
    trailing_pe: float | None = None
    forward_pe: float | None = None
    price_to_book: float | None = None
    price_to_sales: float | None = None
    dividend_yield: float | None = None
    beta: float | None = None

    # Profitability / growth
    profit_margin: float | None = None
    operating_margin: float | None = None
    return_on_equity: float | None = None
    revenue_growth: float | None = None
    earnings_growth: float | None = None
    debt_to_equity: float | None = None

    # Trading-relevance
    fifty_two_week_high: float | None = None
    fifty_two_week_low: float | None = None
    avg_volume_10d: float | None = None

    # Analyst consensus (only when yfinance has it — usually large/mid caps)
    analyst_target_mean: float | None = None
    analyst_recommendation: str | None = None  # e.g. "buy" / "hold" / "sell"
    analyst_count: int | None = None

    # Calendar (next earnings)
    next_earnings_unix: float | None = None

    # One-line summary, capped at 240 chars for /fundamentals + LLM prompt.
    summary: str | None = None

    # Always-present marker: the raw provider name (yfinance for now;
    # leaves the door open for SEC EDGAR / Alpha Vantage / etc.).
    source: str = "yfinance"

    # Free-form extras for fields we don't model explicitly but still
    # want a place to stash without changing the schema. Bounded to
    # keep the cache file small — see :data:`MAX_EXTRA_FIELDS`.
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "FundamentalsSnapshot":
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs: dict[str, Any] = {}
        for k, v in raw.items():
            if k in fields:
                kwargs[k] = v
        if "extras" in kwargs and not isinstance(kwargs["extras"], dict):
            kwargs["extras"] = {}
        if "symbol" not in kwargs or "fetched_at_unix" not in kwargs:
            # Defensive: a malformed row is dropped at load time rather
            # than raising; see CacheLoader._load.
            raise ValueError("missing required field on FundamentalsSnapshot")
        return cls(**kwargs)


MAX_EXTRA_FIELDS = 12


class FundamentalsFetcher(Protocol):
    """Pluggable provider; production implementation is yfinance-backed."""

    name: str

    def fetch(self, symbol: str) -> FundamentalsSnapshot | None:
        """Return a fresh snapshot for ``symbol`` or ``None`` on failure.

        Implementations MUST never raise — log + return ``None`` on
        provider error so the cache layer can decide whether to keep
        serving stale data or skip the symbol entirely.
        """
        ...
