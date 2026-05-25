"""Headline → ticker extraction.

Free-text ticker extraction is famously noisy: regex ``\\b[A-Z]{1,5}\\b``
matches ``A``, ``I``, ``CEO``, ``IPO`` and a thousand other false
positives. We sidestep the problem by using a **dictionary-based**
extractor: a candidate token must exist in a known-symbol vocabulary
before it counts as a ticker.

The vocabulary is built from:

* the eToro :class:`~src.etoro.instrument_cache.InstrumentCache` (every
  ticker the bot has ever resolved); and
* an optional caller-supplied set of "promotable" tickers — typically
  the ones a source has already pre-extracted (StockTwits, yfinance
  ``relatedTickers``, SEC EDGAR after CIK lookup). These tickers are
  always considered known, even if the eToro cache hasn't seen them yet
  in this process.

Common-word safelist
--------------------
A small built-in *stoplist* removes single-letter / common-word symbols
that are technically valid tickers but cause more harm than good in
headline text (``A``, ``I``, ``IT``, ``FOR``, ``ON``, ``BE``, etc.).
Symbols on the stoplist are still extractable when prefixed with ``$``
(StockTwits-style cashtags) — that disambiguates "I" the pronoun from
``$I`` the ticker.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


# Tokens that are technically valid US tickers but appear so often as
# English words / headline filler that bare-form extraction is unsafe.
# We still accept them when preceded by ``$`` (cashtag form).
_STOPLIST: frozenset[str] = frozenset(
    {
        # Single letters
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
        "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
        "U", "V", "W", "X", "Y", "Z",
        # Two-letter common words
        "AI", "AM", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "HE",
        "IF", "IN", "IS", "IT", "ME", "MY", "NO", "OF", "ON", "OR",
        "SO", "TO", "UP", "US", "WE",
        # Three+ letter common words / acronyms that crash with tickers
        "ALL", "AND", "ANY", "ARE", "BUT", "CAN", "CEO", "CFO", "COO",
        "CTO", "DAY", "DID", "EPS", "ETF", "FED", "FOR", "GDP", "GET",
        "HAD", "HAS", "HOT", "IPO", "ITS", "LAW", "LED", "LOW", "MAY",
        "NEW", "NOT", "NOW", "OLD", "ONE", "OUT", "OWN", "PER", "PUT",
        "SEC", "SEE", "SEO", "SET", "TAX", "THE", "TOP", "TWO", "USA",
        "WAS", "WHO", "WHY", "YES", "YOU",
        "MORE", "OPEN", "THIS", "THAT", "WITH", "FROM", "WILL", "WHAT",
        "YEAR", "WEEK", "DOWN", "BACK", "INTO", "OVER", "JUST", "LIKE",
        "MUCH", "NEXT", "ONLY", "SOME", "TIME", "VERY", "WHEN", "STOCK",
        "STOCKS", "PRICE", "EARNINGS", "NEWS",
    }
)


# Cashtag pattern: ``$AAPL`` or ``$btc`` — case-insensitive.
_CASHTAG_RE = re.compile(r"\$([A-Za-z][A-Za-z0-9\.\-]{0,9})\b")

# Bare-form token: 1-5 uppercase letters not preceded by ``$`` (handled
# separately above), possibly with a ``.`` or ``-`` (Berkshire BRK.B, etc.).
_BARE_RE = re.compile(r"(?<![\w\$])([A-Z][A-Z0-9\.\-]{0,4})(?![\w\.])")


@dataclass(frozen=True)
class TickerExtraction:
    """Result of running :meth:`TickerExtractor.extract` on a piece of text.

    ``symbols`` are unique and upper-cased. ``cashtag_only`` is the
    subset that was only recognised via the ``$`` prefix (i.e. would
    have been filtered out by the stoplist). Carrying it separately
    lets downstream code decide whether to trust cashtag-only mentions
    (Reddit / Stocktwits-style) or to require corroboration.
    """

    symbols: tuple[str, ...]
    cashtag_only: tuple[str, ...]

    def __bool__(self) -> bool:
        return bool(self.symbols)


class TickerExtractor:
    """Dictionary-based ticker extractor.

    The vocabulary is *case-insensitive* — tickers are normalised to
    upper-case both on input and on match. Promotion (adding tickers to
    the vocabulary at runtime) is cheap and safe: pass a known set when
    constructing or use :meth:`add_known`.
    """

    def __init__(
        self,
        known_symbols: Iterable[str] = (),
        *,
        stoplist: frozenset[str] = _STOPLIST,
    ) -> None:
        self._known: set[str] = {s.strip().upper() for s in known_symbols if s and s.strip()}
        self._stoplist = stoplist

    def add_known(self, symbols: Iterable[str]) -> None:
        """Add tickers to the recognised vocabulary at runtime."""
        for s in symbols:
            if s and s.strip():
                self._known.add(s.strip().upper())

    @property
    def vocabulary_size(self) -> int:
        return len(self._known)

    def extract(self, text: str) -> TickerExtraction:
        """Return tickers found in ``text``.

        Recognition rules:

        1. ``$XYZ`` cashtags are always candidates (case-insensitive).
        2. Bare-form ``XYZ`` tokens are candidates only when uppercase
           in the source text *and* present in the vocabulary *and* not
           on the stoplist.

        Both groups are filtered against the vocabulary; symbols that
        pass only via cashtag form are recorded in ``cashtag_only`` so
        downstream code can attach a confidence signal.
        """
        if not text:
            return TickerExtraction(symbols=(), cashtag_only=())

        cashtags: set[str] = set()
        for m in _CASHTAG_RE.finditer(text):
            tok = m.group(1).upper()
            if tok in self._known:
                cashtags.add(tok)

        bare: set[str] = set()
        for m in _BARE_RE.finditer(text):
            tok = m.group(1).upper()
            if tok in self._stoplist:
                continue
            if tok in self._known:
                bare.add(tok)

        all_found = bare | cashtags
        cashtag_only = cashtags - bare
        ordered = tuple(sorted(all_found))
        cashtag_only_ordered = tuple(sorted(cashtag_only))
        return TickerExtraction(symbols=ordered, cashtag_only=cashtag_only_ordered)
