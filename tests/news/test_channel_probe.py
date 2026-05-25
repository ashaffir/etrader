"""Tests for the live-probe helper used by /channels test."""

from __future__ import annotations

import unittest
from typing import Any, Iterable

from src.news.channel_probe import probe_many, probe_source
from src.news.sources.base import NewsItem


class _FakeOkSource:
    name = "stocktwits"

    def __init__(self, items: list[NewsItem]) -> None:
        self._items = items
        self.calls: list[dict[str, Any]] = []

    def fetch(self, *, since=None, known_symbols=None) -> Iterable[NewsItem]:
        self.calls.append({"since": since, "known": list(known_symbols or [])})
        return list(self._items)


class _FakeRaisingSource:
    name = "google_news"

    def fetch(self, *, since=None, known_symbols=None):  # noqa: ARG002
        raise RuntimeError("upstream 503")


class _FakeDisabledSource:
    name = "sec_edgar"
    _disabled_reason = "SEC_USER_AGENT not configured"

    def fetch(self, *, since=None, known_symbols=None):  # noqa: ARG002
        raise AssertionError("fetch must not run on a disabled source")


def _item(symbol: str = "AAPL", headline: str = "Apple does a thing") -> NewsItem:
    return NewsItem(
        source="stocktwits",
        symbols=(symbol,),
        headline=headline,
        url=f"https://example.test/{symbol}",
        published_at=1_700_000_000.0,
    )


class ProbeSourceTests(unittest.TestCase):
    def test_ok_source_reports_items_and_sample(self) -> None:
        src = _FakeOkSource([_item("AAPL"), _item("MSFT", "Microsoft news")])
        result = probe_source(src)
        self.assertTrue(result.ok)
        self.assertEqual(result.items_count, 2)
        self.assertEqual(result.sample_headline, "Apple does a thing")
        self.assertIsNone(result.error)
        self.assertIsNone(result.disabled_reason)
        self.assertGreaterEqual(result.duration_ms, 0)

    def test_known_symbols_are_forwarded(self) -> None:
        src = _FakeOkSource([])
        probe_source(src, known_symbols=["AAPL", "TSLA"])
        self.assertEqual(src.calls[-1]["known"], ["AAPL", "TSLA"])

    def test_raising_source_returns_error(self) -> None:
        result = probe_source(_FakeRaisingSource())
        self.assertFalse(result.ok)
        self.assertIsNone(result.disabled_reason)
        self.assertIn("RuntimeError", result.error or "")
        self.assertIn("upstream 503", result.error or "")

    def test_disabled_source_is_short_circuited(self) -> None:
        result = probe_source(_FakeDisabledSource())
        self.assertFalse(result.ok)
        self.assertEqual(result.items_count, 0)
        self.assertEqual(
            result.disabled_reason, "SEC_USER_AGENT not configured",
        )
        self.assertIsNone(result.error)

    def test_long_headline_is_truncated(self) -> None:
        long_text = "x" * 200
        src = _FakeOkSource([_item("AAPL", long_text)])
        result = probe_source(src, headline_max_len=20)
        assert result.sample_headline is not None
        self.assertLessEqual(len(result.sample_headline), 20)
        self.assertTrue(result.sample_headline.endswith("…"))


class ProbeManyTests(unittest.TestCase):
    def test_probes_every_source_in_order(self) -> None:
        results = probe_many([
            _FakeOkSource([_item("AAPL")]),
            _FakeRaisingSource(),
            _FakeDisabledSource(),
        ])
        names = [r.name for r in results]
        self.assertEqual(names, ["stocktwits", "google_news", "sec_edgar"])
        self.assertTrue(results[0].ok)
        self.assertFalse(results[1].ok)
        self.assertFalse(results[2].ok)
        self.assertEqual(
            results[2].disabled_reason, "SEC_USER_AGENT not configured",
        )

    def test_only_filter_is_case_insensitive(self) -> None:
        results = probe_many(
            [_FakeOkSource([_item("AAPL")]), _FakeRaisingSource()],
            only=["GOOGLE_NEWS"],
        )
        self.assertEqual([r.name for r in results], ["google_news"])

    def test_unknown_only_names_are_dropped(self) -> None:
        results = probe_many(
            [_FakeOkSource([_item("AAPL")])],
            only=["nonexistent"],
        )
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
