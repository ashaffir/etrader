"""Tests for the SEC EDGAR 8-K source + CIK→ticker map."""

import json
import tempfile
import time
import unittest
from pathlib import Path

from src.news.sources.sec_edgar import (
    CikTickerMap,
    SecEdgar8KSource,
    _cik_from_title,
)


_FAKE_TICKERS_PAYLOAD = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
    # Malformed row — should be ignored, not raised.
    "2": {"cik_str": "bad", "ticker": "XXX"},
}

_FAKE_FEED = {
    "entries": [
        {
            "title": "8-K - APPLE INC (0000320193) (Filer)",
            "link": "https://sec.gov/aapl-8k",
            "summary": "Item 2.02 results of operations",
            "published_parsed": time.gmtime(time.time() - 300),
        },
        {
            "title": "8-K - UNKNOWN CO (9999999999) (Filer)",
            "link": "https://sec.gov/unknown-8k",
            "summary": "",
            "published_parsed": time.gmtime(time.time() - 600),
        },
        # Missing link → skipped
        {"title": "broken", "summary": ""},
    ]
}


class TitleParserTests(unittest.TestCase):
    def test_extracts_cik(self) -> None:
        self.assertEqual(
            _cik_from_title("8-K - APPLE INC (0000320193) (Filer)"), 320193
        )

    def test_returns_none_for_unparseable(self) -> None:
        self.assertIsNone(_cik_from_title("not a real edgar title"))
        self.assertIsNone(_cik_from_title("8-K - SOMECO (Reporter)"))


class CikTickerMapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp()) / "cik.json"

    def test_update_from_payload_ingests_valid_rows(self) -> None:
        m = CikTickerMap.load(self.tmp)
        count = m.update_from_payload(_FAKE_TICKERS_PAYLOAD)
        self.assertEqual(count, 2)
        self.assertEqual(m.lookup(320193), "AAPL")
        self.assertEqual(m.lookup(789019), "MSFT")

    def test_round_trip_persistence(self) -> None:
        m = CikTickerMap.load(self.tmp)
        m.update_from_payload(_FAKE_TICKERS_PAYLOAD)
        m.save()
        again = CikTickerMap.load(self.tmp)
        self.assertEqual(again.lookup(320193), "AAPL")

    def test_needs_refresh_on_empty(self) -> None:
        m = CikTickerMap.load(self.tmp)
        self.assertTrue(m.needs_refresh())

    def test_corrupt_file_does_not_crash(self) -> None:
        self.tmp.write_text("{not json", encoding="utf-8")
        m = CikTickerMap.load(self.tmp)
        self.assertEqual(m.cik_to_ticker, {})


class SecEdgar8KSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.cache_path = self.tmp / "cik.json"
        # Seed a fresh ticker map so refresh isn't triggered during tests.
        seeded = {
            "cik_to_ticker": {"320193": "AAPL"},
            "last_refresh_unix": time.time(),
        }
        self.cache_path.write_text(json.dumps(seeded), encoding="utf-8")

        self.feed_calls: list[str] = []
        self.ticker_calls: list[str] = []

        def feed_fetcher(url: str) -> dict:
            self.feed_calls.append(url)
            return _FAKE_FEED

        def ticker_fetcher(url: str) -> dict:
            self.ticker_calls.append(url)
            return _FAKE_TICKERS_PAYLOAD

        self.source = SecEdgar8KSource(
            cache_path=self.cache_path,
            feed_fetcher=feed_fetcher,
            tickers_fetcher=ticker_fetcher,
        )

    def test_resolves_known_cik_to_ticker(self) -> None:
        items = list(self.source.fetch())
        # 3 entries — one resolves to AAPL, one to unknown (empty symbols),
        # one is malformed and skipped.
        self.assertEqual(len(items), 2)
        aapl = [it for it in items if it.symbols == ("AAPL",)]
        unknown = [it for it in items if not it.symbols]
        self.assertEqual(len(aapl), 1)
        self.assertEqual(len(unknown), 1)
        self.assertEqual(aapl[0].metadata["form_type"], "8-K")
        self.assertEqual(aapl[0].metadata["cik"], 320193)

    def test_does_not_refresh_when_cache_fresh(self) -> None:
        list(self.source.fetch())
        self.assertEqual(self.ticker_calls, [])

    def test_refreshes_when_cache_stale(self) -> None:
        # Mark map as stale.
        stale = {"cik_to_ticker": {}, "last_refresh_unix": 0.0}
        self.cache_path.write_text(json.dumps(stale), encoding="utf-8")
        source = SecEdgar8KSource(
            cache_path=self.cache_path,
            feed_fetcher=lambda url: _FAKE_FEED,
            tickers_fetcher=lambda url: _FAKE_TICKERS_PAYLOAD,
        )
        list(source.fetch())
        # After fetch, the cache file should now hold 2 rows.
        body = json.loads(self.cache_path.read_text(encoding="utf-8"))
        self.assertEqual(set(body["cik_to_ticker"].keys()), {"320193", "789019"})

    def test_failed_feed_returns_empty(self) -> None:
        def boom(url: str) -> dict:
            raise RuntimeError("sec down")

        source = SecEdgar8KSource(
            cache_path=self.cache_path,
            feed_fetcher=boom,
            tickers_fetcher=lambda url: _FAKE_TICKERS_PAYLOAD,
        )
        self.assertEqual(list(source.fetch()), [])


if __name__ == "__main__":
    unittest.main()
