"""Build a configured news pipeline from :class:`~src.config.AppConfig`.

Keeps source instantiation out of ``main.py`` and out of the cycle.
Both the trading bot and ad-hoc CLIs (e.g. a future ``python -m
src.news.cli scan``) can lean on this single entry-point so the
config → live aggregator wiring lives in exactly one place.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from ..config import NewsConfig
from ..etoro.instrument_cache import InstrumentCache
from .aggregator import NewsAggregator
from .candidate_store import CandidateStore
from .scheduler import NewsScheduler
from .sources.base import NewsSource
from .sources.google_news_rss import DEFAULT_QUERIES, GoogleNewsRssSource
from .sources.sec_edgar import SecEdgar8KSource
from .sources.stocktwits import StockTwitsTrendingSource
from .sources.yahoo_rss import YahooRssSource
from .sources.yfinance_news import YFinanceNewsSource
from .ticker_extractor import TickerExtractor


def build_news_sources(
    cfg: NewsConfig,
    *,
    project_root: Path,
    logger: logging.Logger | logging.LoggerAdapter | None = None,
) -> list[NewsSource]:
    """Instantiate the enabled news sources from config.

    Unknown source names are logged-and-skipped rather than raising,
    so a typo'd ``enabled_sources`` entry never blocks startup.
    """
    log = logger or logging.getLogger("etrader.news.factory")
    base_name = _logger_name(log, fallback="etrader.news.factory")
    out: list[NewsSource] = []
    for name in cfg.enabled_sources:
        canonical = name.strip().lower()
        child_logger = logging.getLogger(f"{base_name}.{canonical}")
        if canonical == "stocktwits":
            out.append(StockTwitsTrendingSource(logger=child_logger))
        elif canonical == "sec_edgar":
            cache_path = project_root / cfg.sec_edgar_cik_cache_path
            out.append(SecEdgar8KSource(cache_path=cache_path, logger=child_logger))
        elif canonical == "google_news":
            queries = cfg.google_news_queries or DEFAULT_QUERIES
            out.append(
                GoogleNewsRssSource(
                    queries=queries,
                    max_items_per_query=cfg.google_news_max_items_per_query,
                    logger=child_logger,
                )
            )
        elif canonical == "yahoo_rss":
            out.append(
                YahooRssSource(
                    max_symbols=cfg.yahoo_rss_max_symbols,
                    max_items_per_symbol=cfg.yahoo_rss_max_items_per_symbol,
                    logger=child_logger,
                )
            )
        elif canonical == "yfinance":
            out.append(
                YFinanceNewsSource(
                    max_symbols=cfg.yfinance_max_symbols,
                    max_items_per_symbol=cfg.yfinance_max_items_per_symbol,
                    logger=child_logger,
                )
            )
        else:
            log.warning("unknown news source %r — skipping", name)
    return out


def build_news_pipeline(
    cfg: NewsConfig,
    *,
    project_root: Path,
    instrument_cache: InstrumentCache,
    extra_known_symbols: Iterable[str] = (),
    logger: logging.Logger | logging.LoggerAdapter | None = None,
) -> tuple[CandidateStore, NewsAggregator, NewsScheduler]:
    """Construct the full news pipeline.

    ``instrument_cache`` seeds the ticker extractor's vocabulary so
    free-text headlines that mention symbols the bot has previously
    resolved are recognised on first scan. ``extra_known_symbols``
    is a hatch for boot-time seeding (e.g. config-supplied seeds).

    Returns ``(store, aggregator, scheduler)`` so the caller can hold
    references it needs — typically ``scheduler`` for the cycle loop,
    ``aggregator`` for ``/news force`` style commands, and ``store``
    for the universe builder.
    """
    log = logger or logging.getLogger("etrader.news.factory")
    base_name = _logger_name(log, fallback="etrader.news.factory")

    store = CandidateStore(
        path=project_root / cfg.candidate_store_path,
        ttl_seconds=cfg.ttl_hours * 3600,
        logger=logging.getLogger(f"{base_name}.store"),
    )

    extractor = TickerExtractor(known_symbols=instrument_cache.known_symbols())
    if extra_known_symbols:
        extractor.add_known(extra_known_symbols)

    sources = build_news_sources(cfg, project_root=project_root, logger=log)
    aggregator = NewsAggregator(
        sources,
        store=store,
        ticker_extractor=extractor,
        half_life_seconds=cfg.half_life_hours * 3600,
        logger=logging.getLogger(f"{base_name}.aggregator"),
    )
    scheduler = NewsScheduler(
        aggregator,
        interval_minutes=cfg.scan_interval_minutes,
        logger=logging.getLogger(f"{base_name}.scheduler"),
    )
    return store, aggregator, scheduler


def _logger_name(
    logger: logging.Logger | logging.LoggerAdapter,
    *,
    fallback: str,
) -> str:
    """Return a ``logging.getLogger``-compatible name from either flavor.

    ``LoggerAdapter`` wraps a real ``Logger`` accessible via ``.logger``;
    ``Logger`` exposes the name on itself. Either way we get a usable
    parent name so child loggers stay in the same hierarchy.
    """
    inner = getattr(logger, "logger", logger)
    name = getattr(inner, "name", None)
    if isinstance(name, str) and name:
        return name
    return fallback
