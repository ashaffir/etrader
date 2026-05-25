"""Convert :class:`FundamentalsSnapshot` instances into LLM-friendly dicts.

The decision prompt is JSON-serialised; we ship a *trim* projection of
each snapshot (only the high-signal valuation / growth / analyst
fields) so the prompt stays well under the model's context budget
even when the universe is large. The full struct is kept in the
cache (and exposed via ``/fundamentals <SYM>``); the LLM doesn't need
``website``, ``floatShares``, and friends.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .types import FundamentalsSnapshot


def project_for_llm(snap: FundamentalsSnapshot) -> dict[str, Any]:
    """Return the prompt-ready dict for one snapshot.

    Returns only fields the LLM should actually weight. ``None``
    values are kept (the model needs to know what's missing) but the
    overall key count is bounded.
    """
    return {
        "symbol": snap.symbol,
        "name": snap.name,
        "sector": snap.sector,
        "industry": snap.industry,
        "quote_type": snap.quote_type,
        "currency": snap.currency,
        "market_cap": snap.market_cap,
        "trailing_pe": snap.trailing_pe,
        "forward_pe": snap.forward_pe,
        "price_to_book": snap.price_to_book,
        "price_to_sales": snap.price_to_sales,
        "dividend_yield": snap.dividend_yield,
        "beta": snap.beta,
        "profit_margin": snap.profit_margin,
        "operating_margin": snap.operating_margin,
        "return_on_equity": snap.return_on_equity,
        "revenue_growth": snap.revenue_growth,
        "earnings_growth": snap.earnings_growth,
        "debt_to_equity": snap.debt_to_equity,
        "fifty_two_week_high": snap.fifty_two_week_high,
        "fifty_two_week_low": snap.fifty_two_week_low,
        "analyst_target_mean": snap.analyst_target_mean,
        "analyst_recommendation": snap.analyst_recommendation,
        "analyst_count": snap.analyst_count,
        "summary": snap.summary,
    }


def build_fundamentals_payload(
    snapshots: Sequence[FundamentalsSnapshot] | Mapping[str, FundamentalsSnapshot],
) -> list[dict[str, Any]]:
    """Project a collection of snapshots into the LLM payload form."""
    if isinstance(snapshots, Mapping):
        iterable = snapshots.values()
    else:
        iterable = snapshots
    return [project_for_llm(s) for s in iterable if s is not None]
