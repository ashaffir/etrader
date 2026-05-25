"""Fundamentals package — per-symbol valuation / growth / analyst data.

Public surface:

- :class:`FundamentalsSnapshot` — the plain-data record consumers see.
- :class:`FundamentalsCache` — persisted symbol→snapshot map with a
  freshness policy (24 h refresh + earnings-aware override).
- :class:`YFinanceFundamentalsFetcher` — production fetcher.
- :func:`build_fundamentals_cache` — factory that wires everything from
  :class:`~src.config.FundamentalsConfig`.

Everything is lazy and zero-cost until something asks for a snapshot:
yfinance is only imported on the first fetch, and missing symbols never
raise — they just resolve to ``None``.
"""

from __future__ import annotations

from .cache import FundamentalsCache
from .factory import build_fundamentals_cache
from .types import FundamentalsFetcher, FundamentalsSnapshot
from .yfinance_fetcher import YFinanceFundamentalsFetcher

__all__ = [
    "FundamentalsCache",
    "FundamentalsFetcher",
    "FundamentalsSnapshot",
    "YFinanceFundamentalsFetcher",
    "build_fundamentals_cache",
]
