"""Tests for the dictionary-based ticker extractor."""

import unittest

from src.news.ticker_extractor import TickerExtractor


class TickerExtractorTests(unittest.TestCase):
    def test_bare_uppercase_token_extracted_when_in_vocabulary(self) -> None:
        ex = TickerExtractor(known_symbols=["AAPL", "MSFT", "NVDA"])
        result = ex.extract("AAPL beats Q3 estimates as MSFT lags behind")
        self.assertEqual(result.symbols, ("AAPL", "MSFT"))
        self.assertEqual(result.cashtag_only, ())

    def test_bare_token_skipped_when_not_in_vocabulary(self) -> None:
        ex = TickerExtractor(known_symbols=["AAPL"])
        result = ex.extract("MSFT and NVDA both rallied")
        self.assertEqual(result.symbols, ())

    def test_cashtag_form_overrides_stoplist(self) -> None:
        # "A" and "I" are technically valid US tickers but stop-listed
        # to avoid false positives. Cashtag form bypasses the stop-list.
        ex = TickerExtractor(known_symbols=["A", "I", "AAPL"])
        result = ex.extract("I think A is overpriced but $A had a great quarter")
        # Bare "A" and "I" are filtered. Cashtag $A is accepted.
        self.assertIn("A", result.symbols)
        self.assertNotIn("I", result.symbols)
        self.assertEqual(result.cashtag_only, ("A",))

    def test_common_english_words_not_extracted_as_tickers(self) -> None:
        ex = TickerExtractor(known_symbols=["FOR", "ON", "THE", "AAPL"])
        # FOR/ON/THE are all stop-listed despite being in the vocabulary.
        result = ex.extract("This is the news FOR AAPL ON the rise")
        self.assertEqual(result.symbols, ("AAPL",))

    def test_case_insensitive_known_symbols(self) -> None:
        ex = TickerExtractor(known_symbols=["aapl"])
        result = ex.extract("$aapl printed an ATH")
        self.assertEqual(result.symbols, ("AAPL",))

    def test_add_known_after_construction(self) -> None:
        ex = TickerExtractor(known_symbols=["AAPL"])
        self.assertFalse(ex.extract("MSFT moved"))
        ex.add_known(["MSFT"])
        self.assertEqual(ex.extract("MSFT moved").symbols, ("MSFT",))

    def test_empty_text_returns_empty(self) -> None:
        ex = TickerExtractor(known_symbols=["AAPL"])
        self.assertEqual(ex.extract("").symbols, ())
        self.assertEqual(ex.extract("   ").symbols, ())

    def test_dotted_tickers_supported(self) -> None:
        ex = TickerExtractor(known_symbols=["BRK.B"])
        result = ex.extract("$BRK.B closes higher")
        self.assertIn("BRK.B", result.symbols)

    def test_extraction_truthiness(self) -> None:
        ex = TickerExtractor(known_symbols=["AAPL"])
        self.assertFalse(ex.extract("no tickers here"))
        self.assertTrue(ex.extract("AAPL won"))


if __name__ == "__main__":
    unittest.main()
