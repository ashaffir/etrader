"""Pricing helpers used by the executor + tests.

Kept separate from :mod:`risk` so the evaluator file stays under the
300-line soft cap and these pure functions can be imported without
pulling in the dataclasses.
"""

from __future__ import annotations

from typing import Iterable

from .risk import TradeVerdict


def compute_stop_loss_take_profit(
    *,
    entry_price: float,
    is_buy: bool,
    stop_loss_pct: float,
    take_profit_pct: float,
) -> tuple[float, float]:
    """Return ``(stop_loss_rate, take_profit_rate)`` aligned with eToro semantics.

    For a BUY (long): SL is below entry, TP is above entry.
    For a SELL (short): SL is above entry, TP is below entry.
    """
    if entry_price <= 0:
        raise ValueError("entry_price must be positive")
    if is_buy:
        sl = entry_price * (1.0 - stop_loss_pct / 100.0)
        tp = entry_price * (1.0 + take_profit_pct / 100.0)
    else:
        sl = entry_price * (1.0 + stop_loss_pct / 100.0)
        tp = entry_price * (1.0 - take_profit_pct / 100.0)
    sl = max(sl, 1e-4)
    tp = max(tp, 1e-4)
    return round(sl, 4), round(tp, 4)


def aggregate_summary(verdicts: Iterable[TradeVerdict]) -> dict[str, int]:
    approved = denied = capped = 0
    for v in verdicts:
        if v.approved:
            approved += 1
            if v.amended_amount_usd is not None:
                capped += 1
        else:
            denied += 1
    return {"approved": approved, "denied": denied, "capped": capped}
