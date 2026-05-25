"""Interval-gated wrapper around :class:`NewsAggregator`.

The aggregator itself is "run once on demand". The trading bot wants
to call it at a fixed cadence (``news.scan_interval_minutes``) without
re-running every cycle. ``NewsScheduler`` is the tiny stateful piece
that says "yes, run now" or "skip, too soon".

It deliberately keeps no threads / async machinery — the cycle is
already single-threaded and we want to avoid stacking work behind the
trading-loop interpreter lock. Each cycle calls :meth:`maybe_run`
which decides whether to fire the aggregator inline.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Iterable

from .aggregator import AggregatorRunStats, NewsAggregator


class NewsScheduler:
    """Decide when the news aggregator should run.

    Parameters
    ----------
    aggregator:
        The :class:`NewsAggregator` to invoke.
    interval_minutes:
        Minimum number of minutes between runs. Defaults to 60.
    clock:
        Optional monotonic clock for tests.
    logger:
        Optional logger.
    """

    def __init__(
        self,
        aggregator: NewsAggregator,
        *,
        interval_minutes: float = 60.0,
        clock: Callable[[], float] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._aggregator = aggregator
        self._interval_seconds = max(60.0, float(interval_minutes) * 60.0)
        self._clock = clock or time.monotonic
        self._log = logger or logging.getLogger("etrader.news.scheduler")
        self._last_run_monotonic: float | None = None
        self._last_stats: AggregatorRunStats | None = None

    # ------------------------------------------------------------------

    @property
    def last_stats(self) -> AggregatorRunStats | None:
        """Most recent run stats (if any). Read-only — callers must not mutate."""
        return self._last_stats

    @property
    def aggregator(self) -> NewsAggregator:
        """Expose the wrapped aggregator (for /news manual triggers)."""
        return self._aggregator

    def seconds_until_next_run(self) -> float:
        """Return the seconds remaining until the next eligible run.

        ``0.0`` means "due now". Useful for logging and tests; not used
        by :meth:`maybe_run` directly.
        """
        if self._last_run_monotonic is None:
            return 0.0
        elapsed = self._clock() - self._last_run_monotonic
        return max(0.0, self._interval_seconds - elapsed)

    def maybe_run(
        self,
        *,
        known_symbols: Iterable[str] | None = None,
        force: bool = False,
    ) -> AggregatorRunStats | None:
        """Run the aggregator if the interval has elapsed (or ``force=True``).

        Returns the stats from the run when one occurred, ``None``
        otherwise. ``known_symbols`` is passed through to per-ticker
        sources (yfinance, Yahoo RSS).
        """
        if not force and self.seconds_until_next_run() > 0:
            return None
        try:
            stats = self._aggregator.run(known_symbols=known_symbols)
        except Exception as exc:  # noqa: BLE001 — never crash the trading loop
            self._log.error("news scan failed: %s", exc)
            return None
        self._last_run_monotonic = self._clock()
        self._last_stats = stats
        self._log.info(
            "[news] scan complete — fetched=%d kept=%d obs=%d took=%.1fs",
            stats.items_fetched,
            stats.items_kept,
            stats.observations_recorded,
            stats.duration_seconds,
        )
        return stats
