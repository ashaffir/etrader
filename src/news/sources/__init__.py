"""News-source plug-ins.

Every source implements the :class:`~src.news.sources.base.NewsSource`
Protocol so the aggregator can fan out without caring about transport
details. All third-party libraries (``yfinance``, ``feedparser``) are
imported lazily inside each source so the package remains importable
even if a dependency is missing — useful for unit tests and partial
deployments.
"""

from __future__ import annotations

from .base import NewsItem, NewsSource

__all__ = ["NewsItem", "NewsSource"]
