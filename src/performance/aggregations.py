"""Pure aggregation helpers for closed-trade records.

Kept dependency-free so they're trivially unit-testable: no IO, no
threading, no clock — pass the data in, get a dict out. The tracker
calls these from inside its lock; the /stats command and /ask LLM
call them indirectly through ``tracker.summary()``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .types import RealizedTrade


def aggregate(trades: list[RealizedTrade]) -> dict[str, Any]:
    """Roll a list of closed trades into a summary dict.

    Keys are the same regardless of input length so callers can render
    a stable table layout for empty windows too.
    """
    if not trades:
        return _empty()
    wins = [t for t in trades if t.realized_pnl_usd > 0]
    losses = [t for t in trades if t.realized_pnl_usd < 0]
    breakeven = [t for t in trades if t.realized_pnl_usd == 0]
    total_realized = sum(t.realized_pnl_usd for t in trades)
    avg_win = (sum(t.realized_pnl_usd for t in wins) / len(wins)) if wins else 0.0
    avg_loss = (sum(t.realized_pnl_usd for t in losses) / len(losses)) if losses else 0.0
    biggest_win = max((t.realized_pnl_usd for t in wins), default=0.0)
    biggest_loss = min((t.realized_pnl_usd for t in losses), default=0.0)
    avg_hold = int(sum(t.hold_seconds for t in trades) / len(trades))
    win_rate = (len(wins) / len(trades)) * 100.0 if trades else 0.0
    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": len(breakeven),
        "win_rate_pct": round(win_rate, 1),
        "realized_pnl_usd": round(total_realized, 2),
        "avg_win_usd": round(avg_win, 2),
        "avg_loss_usd": round(avg_loss, 2),
        "biggest_win_usd": round(biggest_win, 2),
        "biggest_loss_usd": round(biggest_loss, 2),
        "avg_hold_seconds": avg_hold,
    }


def filter_window(
    trades: list[RealizedTrade],
    *,
    now: datetime,
    period: str,
) -> list[RealizedTrade]:
    """Return only trades whose ``closed_at_iso`` falls inside ``period``.

    Supported periods:

    - ``"today"`` — since UTC midnight
    - ``"7d"`` / ``"30d"`` — rolling N-day window
    - ``"all"`` — no filter
    """
    if period == "all":
        return list(trades)
    if period == "today":
        cutoff = (now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    elif period == "7d":
        cutoff = 7 * 24 * 3600
    elif period == "30d":
        cutoff = 30 * 24 * 3600
    else:
        return list(trades)
    return [
        t for t in trades
        if _seconds_since(_parse_iso(t.closed_at_iso), now) <= cutoff
    ]


def by_symbol(trades: list[RealizedTrade]) -> list[dict[str, Any]]:
    """One aggregated row per symbol, sorted by P/L descending."""
    buckets: dict[str, list[RealizedTrade]] = {}
    for t in trades:
        buckets.setdefault(t.symbol or "?", []).append(t)
    rows = [{"symbol": sym, **aggregate(items)} for sym, items in buckets.items()]
    rows.sort(key=lambda r: r["realized_pnl_usd"], reverse=True)
    return rows


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------

def _empty() -> dict[str, Any]:
    return {
        "trades": 0, "wins": 0, "losses": 0, "breakeven": 0,
        "win_rate_pct": 0.0,
        "realized_pnl_usd": 0.0,
        "avg_win_usd": 0.0,
        "avg_loss_usd": 0.0,
        "biggest_win_usd": 0.0,
        "biggest_loss_usd": 0.0,
        "avg_hold_seconds": 0,
    }


def _parse_iso(iso: str) -> datetime:
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return datetime.now(timezone.utc)


def _seconds_since(ts: datetime, now: datetime) -> float:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return max(0.0, (now - ts).total_seconds())
