"""Universe + instrument-cache tests."""

import json
import tempfile
import unittest
from pathlib import Path

from src.etoro.instrument_cache import InstrumentCache


class InstrumentCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp()) / "cache.json"

    def test_round_trip_roundtrips(self) -> None:
        cache = InstrumentCache.load(self.tmp)
        cache.upsert("aapl", 1001)
        cache.upsert("MSFT", 1002)
        cache.save()
        # Reload
        again = InstrumentCache.load(self.tmp)
        self.assertEqual(again.get("AAPL"), 1001)
        self.assertEqual(again.get("aapl"), 1001)  # case-insensitive
        self.assertEqual(again.get("MSFT"), 1002)
        self.assertEqual(again.reverse(1001), "AAPL")

    def test_handles_corrupt_file(self) -> None:
        self.tmp.write_text("{not json", encoding="utf-8")
        cache = InstrumentCache.load(self.tmp)
        self.assertEqual(cache.symbol_to_id, {})

    def test_known_symbols_sorted(self) -> None:
        cache = InstrumentCache.load(self.tmp)
        cache.upsert_many({"ZETA": 9, "alpha": 1, "Mid": 5})
        self.assertEqual(cache.known_symbols(), ["ALPHA", "MID", "ZETA"])

    def test_invalid_id_skipped_on_load(self) -> None:
        self.tmp.write_text(
            json.dumps({"symbol_to_id": {"AAPL": "not-a-number", "MSFT": 1002}}),
            encoding="utf-8",
        )
        cache = InstrumentCache.load(self.tmp)
        self.assertIsNone(cache.get("AAPL"))
        self.assertEqual(cache.get("MSFT"), 1002)


if __name__ == "__main__":
    unittest.main()
