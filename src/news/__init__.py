"""News-driven universe discovery.

This package replaces the bot's previous static ``base_symbols`` list with a
news-driven candidacy pipeline:

* ``sources/`` — one module per free news provider (StockTwits, SEC EDGAR,
  Google News RSS, Yahoo Finance RSS, yfinance ``Ticker.news``). Each one
  emits :class:`~src.news.sources.base.NewsItem` instances.
* :class:`~src.news.ticker_extractor.TickerExtractor` — dictionary-based
  headline → ticker resolution. No NLP / external services; the dictionary
  is built from the eToro instrument cache + any tickers a source has
  pre-extracted (StockTwits / yfinance ``relatedTickers``).
* :class:`~src.news.candidate_store.CandidateStore` — TTL'd, persisted store
  of candidate symbols with attached human-readable reasons (e.g.
  ``"StockTwits trending #3, +8K msgs/24h"``).
* :class:`~src.news.aggregator.NewsAggregator` — fan-out across all enabled
  sources, deduping, scoring, and writing into the candidate store.

The trading bot's existing ``UniverseBuilder`` is rewired in phase 2 to
read from the candidate store instead of hard-coded base symbols.
"""

from __future__ import annotations

from .aggregator import AggregatorRunStats, NewsAggregator
from .candidate_store import Candidate, CandidateStore
from .channel_probe import ChannelProbeResult, probe_many, probe_source
from .factory import build_news_pipeline, build_news_sources
from .scheduler import NewsScheduler
from .sources.base import NewsItem, NewsSource
from .ticker_extractor import TickerExtractor

__all__ = [
    "AggregatorRunStats",
    "Candidate",
    "CandidateStore",
    "ChannelProbeResult",
    "NewsAggregator",
    "NewsItem",
    "NewsScheduler",
    "NewsSource",
    "TickerExtractor",
    "build_news_pipeline",
    "build_news_sources",
    "probe_many",
    "probe_source",
]
