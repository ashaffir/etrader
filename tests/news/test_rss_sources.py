"""Tests for the Google News + Yahoo RSS sources.

Both share the same feedparser-shaped fetcher interface so we test
them together. The fake fetcher returns a hand-built dict matching
``feedparser.parse`` output — no network, no feedparser dependency at
import time.
"""

import time
import unittest
from typing import Any

from src.news.sources.google_news_rss import (
    REGIONAL_LOCALES,
    GoogleNewsLocale,
    GoogleNewsRssSource,
)
from src.news.sources.yahoo_rss import YahooRssSource


def _entry(title: str, link: str, *, seconds_ago: float = 0.0) -> dict[str, Any]:
    """Construct a feedparser-shaped entry with a UTC struct_time."""
    published_ts = time.time() - seconds_ago
    return {
        "title": title,
        "link": link,
        "summary": f"summary for {title}",
        "published_parsed": time.gmtime(published_ts),
    }


class GoogleNewsRssSourceTests(unittest.TestCase):
    def test_fans_out_across_queries(self) -> None:
        seen_urls: list[str] = []

        def fake(url: str) -> dict[str, Any]:
            seen_urls.append(url)
            return {"entries": [_entry(f"hl-{len(seen_urls)}", f"https://x/{len(seen_urls)}")]}

        source = GoogleNewsRssSource(
            queries=["market news today", "earnings beat"],
            fetcher=fake,
        )
        items = list(source.fetch())
        self.assertEqual(len(items), 2)
        # Each query should hit a distinct URL.
        self.assertEqual(len(set(seen_urls)), 2)
        self.assertTrue(any("market+news" in u or "market%20news" in u for u in seen_urls))

    def test_failing_query_does_not_abort(self) -> None:
        def fake(url: str) -> dict[str, Any]:
            if "earnings" in url:
                raise RuntimeError("rate limited")
            return {"entries": [_entry("ok", "https://x/ok")]}

        source = GoogleNewsRssSource(
            queries=["market news", "earnings"], fetcher=fake
        )
        items = list(source.fetch())
        # First query succeeded → 1 item.
        self.assertEqual(len(items), 1)

    def test_since_cutoff_drops_old_entries(self) -> None:
        def fake(url: str) -> dict[str, Any]:
            return {
                "entries": [
                    _entry("recent", "https://x/r", seconds_ago=60),
                    _entry("old", "https://x/o", seconds_ago=3 * 86400),  # 3 days
                ]
            }

        source = GoogleNewsRssSource(queries=["x"], fetcher=fake)
        cutoff = time.time() - 86400  # 1 day ago
        items = list(source.fetch(since=cutoff))
        titles = [it.headline for it in items]
        self.assertIn("recent", titles)
        self.assertNotIn("old", titles)

    def test_limit_per_query(self) -> None:
        def fake(url: str) -> dict[str, Any]:
            return {"entries": [_entry(f"h{i}", f"https://x/{i}") for i in range(50)]}

        source = GoogleNewsRssSource(
            queries=["x"], fetcher=fake, max_items_per_query=10
        )
        items = list(source.fetch())
        self.assertEqual(len(items), 10)

    def test_uk_locale_changes_url_params(self) -> None:
        seen_urls: list[str] = []

        def fake(url: str) -> dict[str, Any]:
            seen_urls.append(url)
            return {"entries": [_entry("UK headline", "https://x/uk")]}

        uk_locale = GoogleNewsLocale(label="UK", hl="en-GB", gl="GB", ceid="GB:en")
        source = GoogleNewsRssSource(
            queries=["FTSE 100"], locale=uk_locale, fetcher=fake,
        )
        items = list(source.fetch())
        self.assertEqual(len(seen_urls), 1)
        # All three locale fields must appear in the URL — that's how
        # Google News routes to the country edition. ``ceid`` contains
        # a literal colon (Google's RSS endpoint accepts it raw,
        # ``quote_plus`` only encodes the query value).
        self.assertIn("hl=en-GB", seen_urls[0])
        self.assertIn("gl=GB", seen_urls[0])
        self.assertIn("ceid=GB:en", seen_urls[0])
        # Emitted item carries the regional source label + locale metadata.
        self.assertEqual(items[0].source, "google_news_uk")
        self.assertEqual(items[0].metadata.get("locale"), "UK")

    def test_regional_locales_cover_expected_markets(self) -> None:
        """Sanity: the starter regional locale set spans UK/EU/Asia/AU.

        If someone trims this tuple by accident, the regression catches
        it before the bot quietly goes back to US-only news. The check
        is a label-membership assertion so reordering / re-labelling the
        tuple is fine as long as each region is represented.
        """
        labels = {loc.label for loc in REGIONAL_LOCALES}
        for required in {"UK", "DE", "JP", "HK", "AU"}:
            self.assertIn(required, labels)


class YahooRssSourceTests(unittest.TestCase):
    def test_requires_known_symbols(self) -> None:
        def fake(url: str) -> dict[str, Any]:
            self.fail("fetcher must not be called when known_symbols is None")

        source = YahooRssSource(fetcher=fake)
        self.assertEqual(list(source.fetch()), [])

    def test_emits_one_item_per_entry_with_symbol_tagged(self) -> None:
        def fake(url: str) -> dict[str, Any]:
            return {"entries": [_entry("AAPL news", "https://x/a")]}

        source = YahooRssSource(fetcher=fake)
        items = list(source.fetch(known_symbols=["aapl"]))
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].symbols, ("AAPL",))
        self.assertEqual(items[0].source, "yahoo_rss")

    def test_dedupes_known_symbols(self) -> None:
        call_count = {"n": 0}

        def fake(url: str) -> dict[str, Any]:
            call_count["n"] += 1
            return {"entries": []}

        source = YahooRssSource(fetcher=fake)
        list(source.fetch(known_symbols=["AAPL", "aapl", "AAPL"]))
        self.assertEqual(call_count["n"], 1)

    def test_max_symbols_cap(self) -> None:
        call_count = {"n": 0}

        def fake(url: str) -> dict[str, Any]:
            call_count["n"] += 1
            return {"entries": []}

        source = YahooRssSource(fetcher=fake, max_symbols=3)
        list(source.fetch(known_symbols=[f"SYM{i}" for i in range(10)]))
        self.assertEqual(call_count["n"], 3)


if __name__ == "__main__":
    unittest.main()
