"""yfinance-backed :class:`FundamentalsFetcher` implementation.

Yfinance is heavy and slow — a single ``Ticker.info`` call can be
500 ms+ and silently hits Yahoo's JSON endpoints. We isolate the call
behind this thin wrapper so:

1. The rest of the bot never imports yfinance at module-load time.
2. Tests can swap in a stub fetcher with zero IO.
3. We get a single place to translate yfinance's wide, inconsistent
   ``info`` dict into our trim :class:`FundamentalsSnapshot`.

The mapping is intentionally conservative: every field we expose is
documented (``yfinance`` deprecates / renames keys between minor
versions). New fields should go through :data:`_FIELD_MAP` so the
test fixture can exercise them.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Mapping

from .types import MAX_EXTRA_FIELDS, FundamentalsFetcher, FundamentalsSnapshot


# (snapshot field, yfinance info key, type-coercer) tuples. We tolerate
# missing keys: most fields are optional and yfinance returns wildly
# different shapes per asset class.
_FIELD_MAP: tuple[tuple[str, str, Callable[[Any], Any]], ...] = (
    ("name",                 "shortName",                       lambda v: str(v).strip() or None),
    ("exchange",             "exchange",                        lambda v: str(v).strip() or None),
    ("quote_type",           "quoteType",                       lambda v: str(v).strip().upper() or None),
    ("sector",               "sector",                          lambda v: str(v).strip() or None),
    ("industry",             "industry",                        lambda v: str(v).strip() or None),
    ("country",              "country",                         lambda v: str(v).strip() or None),
    ("currency",             "currency",                        lambda v: str(v).strip().upper() or None),
    ("market_cap",           "marketCap",                       float),
    ("enterprise_value",     "enterpriseValue",                 float),
    ("trailing_pe",          "trailingPE",                      float),
    ("forward_pe",           "forwardPE",                       float),
    ("price_to_book",        "priceToBook",                     float),
    ("price_to_sales",       "priceToSalesTrailing12Months",    float),
    ("dividend_yield",       "dividendYield",                   float),
    ("beta",                 "beta",                            float),
    ("profit_margin",        "profitMargins",                   float),
    ("operating_margin",     "operatingMargins",                float),
    ("return_on_equity",     "returnOnEquity",                  float),
    ("revenue_growth",       "revenueGrowth",                   float),
    ("earnings_growth",      "earningsGrowth",                  float),
    ("debt_to_equity",       "debtToEquity",                    float),
    ("fifty_two_week_high",  "fiftyTwoWeekHigh",                float),
    ("fifty_two_week_low",   "fiftyTwoWeekLow",                 float),
    ("avg_volume_10d",       "averageVolume10days",             float),
    ("analyst_target_mean",  "targetMeanPrice",                 float),
    ("analyst_recommendation","recommendationKey",              lambda v: str(v).strip().lower() or None),
    ("analyst_count",        "numberOfAnalystOpinions",         int),
    ("next_earnings_unix",   "earningsTimestamp",               float),
)


def _coerce(value: Any, coercer: Callable[[Any], Any]) -> Any:
    """Apply a coercer, returning ``None`` for any failure."""
    if value is None:
        return None
    try:
        out = coercer(value)
    except (TypeError, ValueError):
        return None
    # Filter out yfinance's sentinel-ish values: empty strings & NaN.
    if isinstance(out, float):
        if out != out:  # NaN
            return None
    if isinstance(out, str) and not out:
        return None
    return out


def _trim_summary(raw: Any, *, max_len: int = 240) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if len(s) > max_len:
        s = s[: max_len - 1].rstrip() + "…"
    return s


InfoFetcher = Callable[[str], Mapping[str, Any]]


def _default_info_fetcher(symbol: str) -> Mapping[str, Any]:
    """Default fetcher: lazy-imports yfinance and reads ``Ticker.info``."""
    import yfinance as yf  # noqa: PLC0415 — lazy by design

    info = yf.Ticker(symbol).info or {}
    return info if isinstance(info, Mapping) else {}


class YFinanceFundamentalsFetcher:
    """Production :class:`FundamentalsFetcher` backed by yfinance.

    ``info_fetcher`` is exposed for tests — they can pass a function
    that returns a fixed dict and exercise the full mapping path with
    zero network IO.
    """

    name = "yfinance"

    def __init__(
        self,
        *,
        info_fetcher: InfoFetcher | None = None,
        clock: Callable[[], float] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._fetch_info = info_fetcher or _default_info_fetcher
        self._clock = clock or time.time
        self._log = logger or logging.getLogger("etrader.fundamentals.yfinance")

    def fetch(self, symbol: str) -> FundamentalsSnapshot | None:
        sym = (symbol or "").strip().upper()
        if not sym:
            return None
        try:
            info = self._fetch_info(sym)
        except Exception as exc:  # noqa: BLE001 — never propagate provider failures
            self._log.warning("[fundamentals] yfinance fetch failed for %s: %s", sym, exc)
            return None
        if not info:
            return None
        return self._from_info(sym, info)

    def _from_info(self, symbol: str, info: Mapping[str, Any]) -> FundamentalsSnapshot:
        kwargs: dict[str, Any] = {}
        for field_name, info_key, coercer in _FIELD_MAP:
            kwargs[field_name] = _coerce(info.get(info_key), coercer)
        # Long business summary is sometimes ~1 KB; we cap it so the
        # cache file and LLM prompt stay readable.
        kwargs["summary"] = _trim_summary(info.get("longBusinessSummary"))
        # Bundle a small "extras" payload for fields we don't model
        # but want available to advanced operators via /fundamentals.
        extras: dict[str, Any] = {}
        for k in ("website", "fullTimeEmployees", "sharesOutstanding", "floatShares",
                  "heldPercentInsiders", "heldPercentInstitutions"):
            v = info.get(k)
            if v is not None:
                extras[k] = v
                if len(extras) >= MAX_EXTRA_FIELDS:
                    break
        kwargs["extras"] = extras
        return FundamentalsSnapshot(
            symbol=symbol,
            fetched_at_unix=float(self._clock()),
            **kwargs,
        )
