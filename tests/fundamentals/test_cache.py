"""Unit tests for :class:`FundamentalsCache`.

Exercises the freshness policy, the refresh budget cap, the
stop-event short-circuit, and JSON round-trip persistence.
"""

import json
import tempfile
import unittest
from pathlib import Path

from src.fundamentals.cache import FundamentalsCache
from src.fundamentals.types import FundamentalsSnapshot


class _StubFetcher:
    name = "stub"

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.fail_for: set[str] = set()
        self.next_fetched_at = 1_000.0

    def fetch(self, symbol: str):
        self.calls.append(symbol)
        if symbol in self.fail_for:
            return None
        return FundamentalsSnapshot(
            symbol=symbol,
            fetched_at_unix=self.next_fetched_at,
            name=f"{symbol} Co",
            sector="Tech",
            trailing_pe=20.0,
        )


def _make_cache(tmpdir: Path, *, clock=None, fetcher: _StubFetcher | None = None) -> FundamentalsCache:
    return FundamentalsCache(
        fetcher=fetcher or _StubFetcher(),
        path=tmpdir / "fundamentals.json",
        refresh_after_hours=1.0,    # 1 hour
        failure_backoff_hours=1.0,  # 1 hour
        clock=clock or (lambda: 1_000.0),
    )


class FreshnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmpdir = Path(self._tmp.name)

    def test_missing_symbol_is_stale(self) -> None:
        cache = _make_cache(self.tmpdir)
        self.assertTrue(cache.is_stale("AAPL"))

    def test_recent_entry_not_stale(self) -> None:
        fetcher = _StubFetcher()
        fetcher.next_fetched_at = 1_000.0
        cache = _make_cache(self.tmpdir, clock=lambda: 1_000.0, fetcher=fetcher)
        cache.refresh(["AAPL"])
        self.assertFalse(cache.is_stale("AAPL"))

    def test_entry_stale_after_refresh_window(self) -> None:
        fetcher = _StubFetcher()
        clock_time = [1_000.0]
        cache = _make_cache(self.tmpdir, clock=lambda: clock_time[0], fetcher=fetcher)
        fetcher.next_fetched_at = clock_time[0]
        cache.refresh(["AAPL"])
        # Jump forward beyond the 1 h refresh window.
        clock_time[0] += 3_600 + 60
        self.assertTrue(cache.is_stale("AAPL"))

    def test_earnings_passed_makes_stale(self) -> None:
        fetcher = _StubFetcher()
        clock_time = [1_000.0]
        cache = _make_cache(self.tmpdir, clock=lambda: clock_time[0], fetcher=fetcher)

        # Inject a snapshot whose earnings timestamp has already passed.
        def fetch_with_earnings(sym: str):
            return FundamentalsSnapshot(
                symbol=sym,
                fetched_at_unix=clock_time[0],
                next_earnings_unix=clock_time[0] - 60,  # 60s in the past
            )

        fetcher.fetch = fetch_with_earnings  # type: ignore[assignment]
        cache.refresh(["AAPL"])
        # Still well inside the 1 h window — but earnings already passed.
        self.assertTrue(cache.is_stale("AAPL"))


class RefreshTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmpdir = Path(self._tmp.name)

    def test_budget_caps_calls(self) -> None:
        fetcher = _StubFetcher()
        cache = _make_cache(self.tmpdir, fetcher=fetcher)
        results = cache.refresh(["AAPL", "MSFT", "NVDA", "GOOG"], budget=2)
        refreshed = [k for k, v in results.items() if v == "refreshed"]
        skipped = [k for k, v in results.items() if v == "skipped"]
        self.assertEqual(len(refreshed), 2)
        self.assertEqual(len(skipped), 2)
        # Order is preserved: first two are the refreshed ones.
        self.assertEqual(refreshed, ["AAPL", "MSFT"])

    def test_failure_results_in_backoff(self) -> None:
        fetcher = _StubFetcher()
        fetcher.fail_for = {"AAPL"}
        clock_time = [1_000.0]
        cache = _make_cache(self.tmpdir, clock=lambda: clock_time[0], fetcher=fetcher)
        result = cache.refresh(["AAPL"])
        self.assertEqual(result["AAPL"], "failed")
        # Even though missing entry would normally be stale, the backoff
        # marks it not-stale yet.
        self.assertFalse(cache.is_stale("AAPL"))
        # After the backoff elapses, it becomes refreshable again.
        clock_time[0] += 3_600 + 60
        self.assertTrue(cache.is_stale("AAPL"))

    def test_stop_event_short_circuits(self) -> None:
        fetcher = _StubFetcher()
        cache = _make_cache(self.tmpdir, fetcher=fetcher)
        # Stop event "fires" after the first fetch.
        fired = [False]
        def is_stopping() -> bool:
            return fired[0]

        # Intercept fetch so we flip the stop flag mid-batch.
        original = fetcher.fetch
        def wrapped(sym: str):
            snap = original(sym)
            fired[0] = True
            return snap
        fetcher.fetch = wrapped  # type: ignore[assignment]

        results = cache.refresh(["A", "B", "C"], is_stopping=is_stopping)
        refreshed = [k for k, v in results.items() if v == "refreshed"]
        skipped = [k for k, v in results.items() if v == "skipped"]
        self.assertEqual(refreshed, ["A"])
        self.assertEqual(set(skipped), {"B", "C"})

    def test_unchanged_when_not_stale(self) -> None:
        fetcher = _StubFetcher()
        cache = _make_cache(self.tmpdir, fetcher=fetcher)
        cache.refresh(["AAPL"])
        fetcher.calls.clear()
        results = cache.refresh(["AAPL"])
        self.assertEqual(results["AAPL"], "unchanged")
        self.assertEqual(fetcher.calls, [])


class PersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmpdir = Path(self._tmp.name)

    def test_round_trip(self) -> None:
        fetcher = _StubFetcher()
        cache_path = self.tmpdir / "fundamentals.json"
        cache = FundamentalsCache(
            fetcher=fetcher,
            path=cache_path,
            refresh_after_hours=24.0,
            clock=lambda: 1_000.0,
        )
        cache.refresh(["AAPL", "MSFT"])
        self.assertTrue(cache_path.exists())
        body = json.loads(cache_path.read_text())
        self.assertIn("AAPL", body["items"])
        self.assertEqual(body["items"]["AAPL"]["name"], "AAPL Co")
        # Reopen and verify a fresh instance loads everything back.
        reopened = FundamentalsCache(
            fetcher=_StubFetcher(),  # different fetcher; load is from disk
            path=cache_path,
            refresh_after_hours=24.0,
            clock=lambda: 1_000.0,
        )
        self.assertEqual(len(reopened), 2)
        self.assertEqual(reopened.get("AAPL").name, "AAPL Co")  # type: ignore[union-attr]

    def test_corrupt_file_is_tolerated(self) -> None:
        cache_path = self.tmpdir / "fundamentals.json"
        cache_path.write_text("{ not json")
        cache = FundamentalsCache(
            fetcher=_StubFetcher(),
            path=cache_path,
            refresh_after_hours=24.0,
        )
        self.assertEqual(len(cache), 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
