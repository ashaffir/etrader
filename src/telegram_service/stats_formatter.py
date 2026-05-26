"""Text rendering for the /stats command.

One function per inline-keyboard view; each takes the structured
payload returned by the control API and returns a compact monospace
string suitable for a Telegram message. Keep the column widths under
the smallest sensible phone width (~32 chars) so the tables don't
wrap mid-row.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def format_disabled(payload: Mapping[str, Any]) -> str:
    return (
        "[STATS] Performance tracker not configured.\n"
        "Start the bot with the performance subsystem enabled "
        "and try again."
    )


# ----------------------------------------------------------------------
# Top-level
# ----------------------------------------------------------------------

def format_overview(payload: Mapping[str, Any]) -> str:
    if not payload.get("enabled", True):
        return format_disabled(payload)
    bot = payload.get("bot") or {}
    account = payload.get("account") or {}
    by_period = payload.get("by_period") or {}
    today = by_period.get("today") or {}
    week = by_period.get("7d") or {}
    all_t = by_period.get("all") or {}
    lines = [
        "[STATS — OVERVIEW]",
        f"Open bot positions:   {bot.get('open_position_count', 0)}",
        f"Bot unrealized P/L:   {_money(bot.get('unrealized_pnl_usd'))}",
        f"Account unrealized:   {_money(account.get('unrealized_pnl_usd'))}",
        "",
        f"Trades today:         {today.get('trades', 0)}  "
        f"({today.get('wins', 0)}W / {today.get('losses', 0)}L)",
        f"Realized today:       {_money(today.get('realized_pnl_usd'))}",
        f"Realized last 7d:     {_money(week.get('realized_pnl_usd'))}  "
        f"({week.get('trades', 0)} trades, win-rate {week.get('win_rate_pct', 0):.0f}%)",
        f"Realized all-time:    {_money(all_t.get('realized_pnl_usd'))}  "
        f"({all_t.get('trades', 0)} trades, win-rate {all_t.get('win_rate_pct', 0):.0f}%)",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Window views (today / 7d / 30d / all)
# ----------------------------------------------------------------------

def format_window(payload: Mapping[str, Any], *, period: str) -> str:
    if not payload.get("enabled", True):
        return format_disabled(payload)
    by_period = payload.get("by_period") or {}
    agg = by_period.get(period) or {}
    label = {"today": "TODAY", "7d": "LAST 7 DAYS", "30d": "LAST 30 DAYS", "all": "ALL-TIME"}.get(
        period, period.upper()
    )
    lines = [
        f"[STATS — {label}]",
        f"Trades:           {agg.get('trades', 0)}",
        f"Wins / Losses:    {agg.get('wins', 0)} / {agg.get('losses', 0)}  "
        f"(BE {agg.get('breakeven', 0)})",
        f"Win rate:         {agg.get('win_rate_pct', 0):.1f}%",
        f"Realized P/L:     {_money(agg.get('realized_pnl_usd'))}",
        f"Avg win / loss:   {_money(agg.get('avg_win_usd'))} / "
        f"{_money(agg.get('avg_loss_usd'))}",
        f"Biggest win:      {_money(agg.get('biggest_win_usd'))}",
        f"Biggest loss:     {_money(agg.get('biggest_loss_usd'))}",
        f"Avg hold:         {_pretty_seconds(agg.get('avg_hold_seconds') or 0)}",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Open / closed / by-symbol / daily
# ----------------------------------------------------------------------

def format_open(payload: Mapping[str, Any]) -> str:
    if not payload.get("enabled", True):
        return format_disabled(payload)
    rows = payload.get("open") or []
    if not rows:
        return "[STATS — OPEN]\nThe bot currently has no open positions."
    lines = ["[STATS — OPEN]"]
    lines.append(f"{'sym':<8} {'amt':>7} {'P/L':>8} {'P/L%':>7}  MFE/MAE")
    for r in rows:
        lines.append(
            f"{(r.get('symbol') or '?')[:8]:<8} "
            f"{_money_short(r.get('amount_usd')):>7} "
            f"{_money_short(r.get('last_pnl_usd')):>8} "
            f"{_pct(r.get('last_pnl_pct')):>7}  "
            f"{_money_short(r.get('mfe_usd'))} / "
            f"{_money_short(r.get('mae_usd'))}"
        )
    return "\n".join(lines)


def format_closed(payload: Mapping[str, Any]) -> str:
    if not payload.get("enabled", True):
        return format_disabled(payload)
    rows = payload.get("rows") or []
    if not rows:
        return "[STATS — CLOSED]\nNo bot-closed trades on record yet."
    lines = ["[STATS — CLOSED] (newest last)"]
    lines.append(f"{'date':<10} {'sym':<6} {'P/L':>8} {'%':>6}  hold")
    for r in rows[-20:]:
        closed = (r.get("closed_at_iso") or "")[:10]
        sym = (r.get("symbol") or "?")[:6]
        pnl = _money_short(r.get("realized_pnl_usd"))
        pct = _pct(r.get("realized_pnl_pct"))
        hold = _pretty_seconds(r.get("hold_seconds") or 0)
        lines.append(f"{closed:<10} {sym:<6} {pnl:>8} {pct:>6}  {hold}")
    return "\n".join(lines)


def format_by_symbol(payload: Mapping[str, Any]) -> str:
    if not payload.get("enabled", True):
        return format_disabled(payload)
    rows: Sequence[Mapping[str, Any]] = payload.get("rows") or []
    if not rows:
        return "[STATS — BY SYMBOL]\nNo closed bot trades yet."
    lines = ["[STATS — BY SYMBOL]"]
    lines.append(f"{'sym':<8} {'n':>3} {'win%':>5} {'P/L':>9}")
    for r in rows[:20]:
        lines.append(
            f"{(r.get('symbol') or '?')[:8]:<8} "
            f"{int(r.get('trades') or 0):>3} "
            f"{float(r.get('win_rate_pct') or 0):>4.0f}% "
            f"{_money_short(r.get('realized_pnl_usd')):>9}"
        )
    return "\n".join(lines)


def format_daily(payload: Mapping[str, Any]) -> str:
    if not payload.get("enabled", True):
        return format_disabled(payload)
    rows: Sequence[Mapping[str, Any]] = payload.get("rows") or []
    if not rows:
        return "[STATS — DAILY]\nNo daily snapshots on record yet."
    lines = ["[STATS — DAILY] (newest last)"]
    lines.append(f"{'date':<10} {'equity':>10} {'bot P/L':>9} {'trades':>6}")
    for r in rows[-14:]:
        date = (r.get("date_iso") or "")[:10]
        equity = _money_short(r.get("equity_close"))
        bot_pnl = _money_short(r.get("bot_unrealized_close_usd"))
        trades = int(r.get("bot_trades_today") or 0)
        lines.append(f"{date:<10} {equity:>10} {bot_pnl:>9} {trades:>6}")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------

def _money(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"${float(v):,.2f}"
    except (TypeError, ValueError):
        return "—"


def _money_short(v: Any) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    sign = "-" if f < 0 else " "
    return f"{sign}${abs(f):,.2f}"


def _pct(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):+.1f}%"
    except (TypeError, ValueError):
        return "—"


def _pretty_seconds(s: int | float) -> str:
    try:
        secs = int(s)
    except (TypeError, ValueError):
        return "—"
    if secs <= 0:
        return "0s"
    days, secs = divmod(secs, 86400)
    hours, secs = divmod(secs, 3600)
    minutes, secs = divmod(secs, 60)
    if days:
        return f"{days}d{hours}h"
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"
