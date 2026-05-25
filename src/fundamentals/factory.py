"""Wire a :class:`FundamentalsCache` from :class:`AppConfig`.

Kept separate from the cache class so production wiring stays out of
the test path. Tests construct caches with stub fetchers directly.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .cache import FundamentalsCache
from .types import FundamentalsFetcher
from .yfinance_fetcher import YFinanceFundamentalsFetcher


def build_fundamentals_cache(
    cfg,  # FundamentalsConfig — typed loosely to avoid a circular import
    *,
    project_root: Path,
    fetcher: FundamentalsFetcher | None = None,
    logger: logging.Logger | logging.LoggerAdapter | None = None,
) -> FundamentalsCache | None:
    """Build the cache, or return ``None`` when the feature is disabled.

    The caller (``src.main``) treats a ``None`` cache as "no
    fundamentals enrichment this run" — the cycle, controller and
    Telegram surfaces all degrade gracefully.
    """
    if not cfg.enabled:
        return None
    base = logger or logging.getLogger("etrader.fundamentals")
    # Mirror the pattern used by the news factory: keep a child logger
    # per subcomponent for ``--log-level fundamentals=DEBUG`` workflows.
    inner = getattr(base, "logger", base)
    base_name = getattr(inner, "name", "etrader.fundamentals")
    fetcher = fetcher or YFinanceFundamentalsFetcher(
        logger=logging.getLogger(f"{base_name}.yfinance"),
    )
    return FundamentalsCache(
        fetcher=fetcher,
        path=project_root / cfg.cache_path,
        refresh_after_hours=cfg.refresh_after_hours,
        failure_backoff_hours=cfg.failure_backoff_hours,
        logger=logging.getLogger(f"{base_name}.cache"),
    )
