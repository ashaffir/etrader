"""Asset-class classification regression tests.

eToro's ``/market-data/instruments`` returns ``instrumentTypeID = 5``
for cash equities (AMD, JPM, LUNR, …) and ``instrumentTypeID = 10`` for
crypto (BTC, BCH, XRPAUD). The bot's original mapping had the opposite
assumption, which caused ``asset_class_for`` to label every stock in
the user's account as CRYPTO. That misclassification then:

- made the bot skip the market-hours gate on stocks (crypto is 24/7);
- caused the stuck-order canceller to nuke real stock orders 5 min
  after placement (because for "crypto" the session is always open
  and the order should fill immediately);
- and broke the leverage cap (``is_crypto`` returned True for stocks).

These tests pin the new, evidence-based classifier in place. The raw
payloads below are real samples from the live eToro API (truncated to
the fields the classifier looks at).
"""

import unittest

from src.etoro.market_data import InstrumentMeta
from src.strategy.tools.base import AssetClass, asset_class_for


def _meta(**fields) -> InstrumentMeta:
    raw = {
        "instrumentID": fields.get("instrument_id", 0),
        "instrumentDisplayName": fields.get("display_name"),
        "symbolFull": fields.get("symbol_full"),
        "instrumentTypeID": fields.get("instrument_type_id"),
        "exchangeID": fields.get("exchange_id"),
        "priceSource": fields.get("price_source"),
        "stocksIndustryID": fields.get("stocks_industry_id"),
    }
    return InstrumentMeta(
        instrument_id=int(fields.get("instrument_id") or 0),
        display_name=fields.get("display_name"),
        symbol_full=fields.get("symbol_full"),
        instrument_type_id=fields.get("instrument_type_id"),
        exchange_id=fields.get("exchange_id"),
        raw=raw,
    )


class StockClassificationTests(unittest.TestCase):
    """Real eToro payloads — all of these must classify as STOCK."""

    def test_amd_classifies_as_stock(self) -> None:
        m = _meta(
            instrument_id=1832, symbol_full="AMD",
            display_name="Advanced Micro Devices Inc",
            instrument_type_id=5, exchange_id=4,
            price_source="NASDAQ", stocks_industry_id=8,
        )
        self.assertEqual(asset_class_for(m), AssetClass.STOCK)

    def test_lunr_classifies_as_stock(self) -> None:
        """The whole reason this test file exists."""
        m = _meta(
            instrument_id=10967, symbol_full="LUNR",
            display_name="Intuitive Machines Inc",
            instrument_type_id=5, exchange_id=4,
            price_source="NASDAQ", stocks_industry_id=6,
        )
        self.assertEqual(asset_class_for(m), AssetClass.STOCK)

    def test_jpm_classifies_as_stock(self) -> None:
        m = _meta(
            instrument_id=1023, symbol_full="JPM",
            display_name="JPMorgan Chase & Co",
            instrument_type_id=5, exchange_id=5,
            price_source="NYSE", stocks_industry_id=14,
        )
        self.assertEqual(asset_class_for(m), AssetClass.STOCK)


class CryptoClassificationTests(unittest.TestCase):
    def test_btc_classifies_as_crypto(self) -> None:
        m = _meta(
            instrument_id=100000, symbol_full="BTC",
            display_name="Bitcoin", instrument_type_id=10,
            exchange_id=8, price_source="eToro",
            stocks_industry_id=None,
        )
        self.assertEqual(asset_class_for(m), AssetClass.CRYPTO)
        self.assertTrue(m.is_crypto)

    def test_xrp_classifies_as_crypto(self) -> None:
        m = _meta(
            instrument_id=100163, symbol_full="XRPAUD",
            display_name="XRP / Australian Dollar", instrument_type_id=10,
            exchange_id=8, price_source="eToro",
            stocks_industry_id=0,  # crypto sometimes ships 0, not None
        )
        self.assertEqual(asset_class_for(m), AssetClass.CRYPTO)


class FallbackTests(unittest.TestCase):
    def test_unknown_meta_falls_back_to_symbol(self) -> None:
        m = _meta(
            instrument_id=1, symbol_full="ETH",
            instrument_type_id=None, price_source=None,
            stocks_industry_id=None,
        )
        self.assertEqual(asset_class_for(m), AssetClass.CRYPTO)

    def test_unknown_everything_is_other(self) -> None:
        m = _meta(instrument_id=1, symbol_full="MYSTERY")
        self.assertEqual(asset_class_for(m), AssetClass.OTHER)

    def test_index_symbol_classified_correctly(self) -> None:
        m = _meta(
            instrument_id=1, symbol_full="SPX500",
            instrument_type_id=None, price_source=None,
        )
        self.assertEqual(asset_class_for(m), AssetClass.INDEX)

    def test_typeid_fallback_only_when_no_other_signal(self) -> None:
        """Pure typeID=5 with no priceSource still resolves to STOCK
        (because that's what eToro empirically means by 5)."""
        m = _meta(
            instrument_id=1, symbol_full="???",
            instrument_type_id=5, price_source=None,
            stocks_industry_id=None,
        )
        self.assertEqual(asset_class_for(m), AssetClass.STOCK)

    def test_price_source_overrides_typeid(self) -> None:
        """Even if typeID is suspicious, an equity priceSource wins."""
        m = _meta(
            instrument_id=1, symbol_full="???",
            instrument_type_id=99, price_source="NASDAQ",
            stocks_industry_id=10,
        )
        self.assertEqual(asset_class_for(m), AssetClass.STOCK)


class IsCryptoTests(unittest.TestCase):
    """The ``is_crypto`` heuristic on ``InstrumentMeta`` gates leverage
    rules elsewhere; making sure it agrees with the classifier."""

    def test_stock_is_not_crypto(self) -> None:
        m = _meta(
            symbol_full="AMD", instrument_type_id=5,
            price_source="NASDAQ", stocks_industry_id=8,
        )
        self.assertFalse(m.is_crypto)

    def test_crypto_is_crypto(self) -> None:
        m = _meta(
            symbol_full="BTC", instrument_type_id=10,
            price_source="eToro",
        )
        self.assertTrue(m.is_crypto)


if __name__ == "__main__":
    unittest.main()
