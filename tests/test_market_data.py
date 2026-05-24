"""Tests for the market-data parsers.

Specifically pins down the candle-null-skip behavior: eToro returns
null OHLCV fields for thin / inactive periods, and the parser must
silently drop those candles rather than letting ``None`` flow into
``float()`` and crash the cycle.
"""

import unittest

from src.etoro.market_data import LiveRate, fetch_candles, fetch_rates, _rate_from


class _ClientStub:
    """Minimal eToro client double — returns a fixed payload for any GET."""

    def __init__(self, payload):
        self.payload = payload
        self.last_path = None
        self.last_params = None

    def get(self, path, params=None, retries=0):  # noqa: ARG002
        self.last_path = path
        self.last_params = params
        return self.payload


class CandleParserTests(unittest.TestCase):
    def test_skips_candles_with_null_ohlc(self) -> None:
        payload = {
            "interval": "OneHour",
            "candles": [
                {
                    "instrumentId": 100000,
                    "candles": [
                        # valid
                        {"instrumentID": 100000, "fromDate": "t1",
                         "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05,
                         "volume": 0},
                        # null close — must be skipped
                        {"instrumentID": 100000, "fromDate": "t2",
                         "open": 1.0, "high": 1.1, "low": 0.9, "close": None,
                         "volume": 0},
                        # null open — must be skipped
                        {"instrumentID": 100000, "fromDate": "t3",
                         "open": None, "high": 1.1, "low": 0.9, "close": 1.0,
                         "volume": 0},
                        # null volume only — kept (volume defaults to 0)
                        {"instrumentID": 100000, "fromDate": "t4",
                         "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05,
                         "volume": None},
                        # zero close — skipped
                        {"instrumentID": 100000, "fromDate": "t5",
                         "open": 1.0, "high": 1.1, "low": 0.9, "close": 0,
                         "volume": 1},
                    ],
                }
            ],
        }
        candles = fetch_candles(_ClientStub(payload), 100000)
        self.assertEqual(len(candles), 2)
        self.assertEqual(candles[0].from_date, "t1")
        self.assertEqual(candles[1].from_date, "t4")
        self.assertEqual(candles[1].volume, 0.0)

    def test_handles_empty_payload(self) -> None:
        self.assertEqual(fetch_candles(_ClientStub({}), 1), [])
        self.assertEqual(fetch_candles(_ClientStub({"candles": []}), 1), [])

    def test_count_clamped_to_max(self) -> None:
        # Sanity: passing a value above 1000 should silently cap to 1000 in the URL.
        client = _ClientStub({"candles": []})
        fetch_candles(client, 42, count=10_000, interval="OneDay", direction="desc")
        self.assertIn("/desc/OneDay/1000", client.last_path)


class RatesParserTests(unittest.TestCase):
    def test_rate_from_handles_null_fields(self) -> None:
        rate = _rate_from({
            "instrumentID": 5,
            "ask": None,
            "bid": None,
            "lastExecution": 100.5,
        })
        self.assertEqual(rate.instrument_id, 5)
        self.assertIsNone(rate.ask)
        self.assertIsNone(rate.bid)
        self.assertEqual(rate.last, 100.5)
        # mid falls back to last when ask/bid missing
        self.assertEqual(rate.mid, 100.5)

    def test_fetch_rates_chunks_at_100(self) -> None:
        # Build 150 IDs; the function should chunk into two calls (100 + 50).
        client = _ClientStub({"rates": []})
        ids = list(range(1, 151))
        out = fetch_rates(client, ids)
        # Only the LAST call's params are saved; verify call shape on it.
        self.assertEqual(client.last_path, "/market-data/instruments/rates")
        self.assertIsNotNone(client.last_params)
        # On the last (smaller) call there should be 50 IDs.
        self.assertEqual(len(client.last_params["instrumentIds"]), 50)
        self.assertEqual(out, {})


class UrlBuilderTests(unittest.TestCase):
    """eToro rejects %2C in list query params; our client must emit literal commas."""

    def setUp(self) -> None:
        from src.config import EtoroCredentials
        from src.etoro.client import EtoroClient

        creds = EtoroCredentials(public_key="pk", user_key="uk", is_real=False, allow_real=False)
        self.client = EtoroClient(creds)

    def test_list_param_uses_literal_comma(self) -> None:
        url = self.client._build_url(
            "/market-data/instruments", {"instrumentIds": [1, 2, 3]},
        )
        self.assertIn("instrumentIds=1,2,3", url)
        self.assertNotIn("%2C", url)

    def test_comma_joined_string_preserved(self) -> None:
        url = self.client._build_url(
            "/market-data/instruments", {"instrumentIds": "1,2,3"},
        )
        self.assertIn("instrumentIds=1,2,3", url)
        self.assertNotIn("%2C", url)

    def test_other_special_chars_still_encoded(self) -> None:
        url = self.client._build_url("/x", {"q": "hello world"})
        self.assertIn("q=hello%20world", url)

    def test_none_values_dropped(self) -> None:
        url = self.client._build_url("/x", {"a": "y", "b": None})
        self.assertEqual(url.split("?")[1], "a=y")

    def test_no_params_yields_bare_path(self) -> None:
        self.assertNotIn("?", self.client._build_url("/x", None))
        self.assertNotIn("?", self.client._build_url("/x", {}))


if __name__ == "__main__":
    unittest.main()
