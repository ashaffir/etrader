"""Azure OpenAI per-deployment rate lookup.

The agent records ``usage`` returned by every chat-completion round,
but turning a token count into a dollar figure needs the *rate* the
operator's Azure deployment is billed at. Azure publishes those rates
per model SKU (see https://azure.microsoft.com/pricing/details/azure-openai/),
so we keep a small, hand-curated table of the most common deployments
here.

Matching is by lowercased *prefix* against the deployment name so the
common Azure naming patterns ("gpt-4o", "gpt-4o-2024-08-06",
"gpt-4o-mini-prod-eu") all resolve to the right rate without forcing
the operator to also configure the rate.

Rates are stored in USD per **1,000,000 tokens** (the units Azure
publishes). Callers convert to the smaller per-token figure on demand.

This file is intentionally a tiny static table. If a deployment isn't
listed, :func:`lookup_rates` returns None and the UI shows the cost
estimate as "—" rather than misleading zeros.

When Azure updates a price you only need to edit this file. The pricing
data was captured from the Azure pricing page in May 2026; revisit at
least quarterly. Update the ``_PRICE_TABLE_AS_OF`` constant when you do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

_PRICE_TABLE_AS_OF = "2026-05"


@dataclass(frozen=True)
class TokenRates:
    """USD-per-1M-tokens for one Azure OpenAI deployment family.

    ``cached_per_m`` is the discount rate Azure applies to prompt
    tokens that match its server-side cache prefix. When the SDK
    reports ``prompt_tokens_details.cached_tokens > 0``, those
    tokens are billed at this rate instead of ``input_per_m``.
    Some older / smaller deployments don't offer caching — that's
    represented with ``cached_per_m = None``.

    The :meth:`cost_for` helper produces the dollar figure for a
    single API call given the SDK's token counts.
    """

    family: str
    input_per_m: float
    cached_per_m: float | None
    output_per_m: float

    def cost_for(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
    ) -> float:
        """Return the USD cost of a single chat-completion round.

        Cached prompt tokens are subtracted from the standard-rate
        prompt bucket. When the deployment has no cached rate the
        cached count is treated as standard input (so the dollar
        figure stays honest — under-counting would be misleading).
        """
        prompt = max(0, int(prompt_tokens))
        cached = max(0, int(cached_tokens))
        completion = max(0, int(completion_tokens))
        if self.cached_per_m is None or cached <= 0:
            input_billed_standard = prompt
            cached_billed = 0
        else:
            cached_billed = min(cached, prompt)
            input_billed_standard = prompt - cached_billed
        input_cost = (input_billed_standard / 1_000_000) * self.input_per_m
        cached_cost = (
            (cached_billed / 1_000_000) * float(self.cached_per_m)
            if self.cached_per_m is not None else 0.0
        )
        output_cost = (completion / 1_000_000) * self.output_per_m
        return float(input_cost + cached_cost + output_cost)

    def to_dict(self) -> dict[str, float | str | None]:
        return {
            "family": self.family,
            "input_per_m": self.input_per_m,
            "cached_per_m": self.cached_per_m,
            "output_per_m": self.output_per_m,
        }


# ---------------------------------------------------------------------------
# Static price table (Azure pay-as-you-go Global, May 2026)
# ---------------------------------------------------------------------------
#
# Order matters: prefix matching scans top-to-bottom and returns the
# first hit. More-specific prefixes (e.g. "gpt-5-mini") must precede
# their shorter relatives (e.g. "gpt-5") so a "gpt-5-mini-prod"
# deployment is not mis-classified as the flagship GPT-5.
_RATES: tuple[tuple[str, TokenRates], ...] = (
    (
        "gpt-5-nano",
        TokenRates("gpt-5-nano",      0.05, 0.01, 0.40),
    ),
    (
        "gpt-5-mini",
        TokenRates("gpt-5-mini",      0.25, 0.03, 2.00),
    ),
    (
        "gpt-5-codex",
        TokenRates("gpt-5-codex",     1.25, 0.13, 10.00),
    ),
    (
        "gpt-5-pro",
        TokenRates("gpt-5-pro",       15.00, None, 120.00),
    ),
    (
        "gpt-5-chat",
        TokenRates("gpt-5-chat",      1.25, 0.13, 10.00),
    ),
    (
        "gpt-5",
        TokenRates("gpt-5",           1.25, 0.13, 10.00),
    ),
    (
        "gpt-4o-mini",
        TokenRates("gpt-4o-mini",     0.15, 0.075, 0.60),
    ),
    (
        "gpt-4o",
        TokenRates("gpt-4o",          2.50, 1.25, 10.00),
    ),
    (
        "gpt-4.1-mini",
        TokenRates("gpt-4.1-mini",    0.40, 0.10, 1.60),
    ),
    (
        "gpt-4.1",
        TokenRates("gpt-4.1",         2.00, 0.50, 8.00),
    ),
    (
        "o1-mini",
        TokenRates("o1-mini",         3.00, 1.50, 12.00),
    ),
    (
        "o1",
        TokenRates("o1",              15.00, 7.50, 60.00),
    ),
    (
        "o3-mini",
        TokenRates("o3-mini",         1.10, None, 4.40),
    ),
    (
        "o4-mini",
        TokenRates("o4-mini",         1.10, None, 4.40),
    ),
)


def lookup_rates(deployment: str | None) -> TokenRates | None:
    """Return the :class:`TokenRates` matching a deployment name, or None.

    Matching is lowercase-prefix. The first entry in :data:`_RATES`
    that prefixes the (lowercased) deployment wins.
    """
    if not deployment:
        return None
    needle = deployment.strip().lower()
    if not needle:
        return None
    for prefix, rates in _RATES:
        if needle.startswith(prefix):
            return rates
    return None


def known_families() -> Iterable[str]:
    """Return the iterable of family labels we currently know rates for.

    Useful for diagnostics / UI tooltips that list "we know how to
    price these deployments".
    """
    return [r.family for _prefix, r in _RATES]


def price_table_as_of() -> str:
    """Return the YYYY-MM string the static table was last refreshed at."""
    return _PRICE_TABLE_AS_OF


__all__ = [
    "TokenRates",
    "lookup_rates",
    "known_families",
    "price_table_as_of",
]
