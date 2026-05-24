"""Render Telegram replies from control-API JSON.

Pure formatting helpers — no I/O. They're isolated here so the bot
file stays focused on routing and so they're easy to unit-test.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


# Generous emoji-free formatting; ASCII-only fits the existing log style
# and renders consistently across desktop/mobile Telegram clients.

def format_status(status: Mapping[str, Any]) -> str:
    paused = bool(status.get("paused"))
    halted = bool(status.get("halted_today"))
    line1 = "[STATUS]"
    line2 = f"running:    {'no (PAUSED)' if paused else 'yes'}"
    line3 = f"mode:       {status.get('trading_mode', '?')} ({status.get('env_segment', '?')})"
    line4 = f"cycle:      #{status.get('cycle_count', 0)}"
    line5 = f"tracked:    {status.get('tracked_count', 0)} (base={status.get('base_count', 0)}, llm={status.get('llm_count', 0)})"
    line6 = f"bot-owned:  {status.get('bot_owned_position_count', 0)} position(s)"
    line7 = f"halted:     {'YES (kill-switch)' if halted else 'no'}"
    line8 = f"AI:         {'on' if status.get('ai_enabled') else 'off'}"
    last_started = _fmt_unix(status.get("last_cycle_started_unix"))
    last_finished = _fmt_unix(status.get("last_cycle_finished_unix"))
    line9 = f"last cycle: started={last_started}, finished={last_finished}"
    err = status.get("last_error")
    line10 = f"last error: {err}" if err else "last error: none"
    return "\n".join([line1, line2, line3, line4, line5, line6, line7, line8, line9, line10])


def format_portfolio(portfolio: Mapping[str, Any]) -> str:
    summary = portfolio.get("summary") or {}
    positions = portfolio.get("positions") or []
    bot_owned_ids = set(portfolio.get("bot_owned_position_ids") or [])

    lines: list[str] = ["[PORTFOLIO]"]
    lines.append(f"equity:     ${_money(summary.get('equity'))}")
    lines.append(f"available:  ${_money(summary.get('available_cash'))}")
    lines.append(f"invested:   ${_money(summary.get('total_invested'))}")
    lines.append(f"P&L:        ${_money(summary.get('profit_loss'))}")
    lines.append(f"open posns: {len(positions)} ({len(bot_owned_ids)} bot-owned)")
    lines.append("")
    if not positions:
        lines.append("(no open positions)")
        return "\n".join(lines)

    lines.append("symbol  side  amount     pnl       lev  owner")
    for p in positions[:25]:
        symbol = (p.get("symbol") or "?")[:6].ljust(6)
        side = "LONG " if p.get("is_buy") else "SHORT"
        amount = _money(p.get("amount")).rjust(9)
        pnl = _money(p.get("pnl")).rjust(8)
        lev = f"x{int(p.get('leverage') or 1)}".ljust(3)
        if p.get("is_mirror"):
            owner = "mirror"
        elif p.get("is_bot_owned"):
            owner = "bot"
        else:
            owner = "manual"
        lines.append(f"{symbol}  {side}  {amount}  {pnl}  {lev}  {owner}")
    if len(positions) > 25:
        lines.append(f"... and {len(positions) - 25} more")
    return "\n".join(lines)


def format_universe(universe: Mapping[str, Any]) -> str:
    symbols = list(universe.get("symbols") or [])
    base_count = int(universe.get("base_count") or 0)
    llm_count = int(universe.get("llm_count") or 0)
    if not symbols:
        return "[UNIVERSE]\n(no instruments tracked yet — wait for the first cycle)"
    lines = [
        "[UNIVERSE]",
        f"tracking {len(symbols)} instrument(s) (base={base_count}, llm={llm_count})",
        "",
        ", ".join(symbols),
    ]
    return "\n".join(lines)


def format_history(entries: Sequence[Mapping[str, Any]]) -> str:
    if not entries:
        return "[HISTORY]\n(no trades recorded yet)"
    lines = ["[HISTORY] (newest last)"]
    for e in entries:
        ts = e.get("timestamp", "?")
        action = (e.get("action") or "?")[:5].ljust(5)
        status = (e.get("status") or "?")[:11].ljust(11)
        symbol = (e.get("symbol") or "?")[:8].ljust(8)
        amount = e.get("amount_usd")
        amt_str = f"${_money(amount)}" if amount is not None else "—"
        lines.append(f"{ts}  {action}  {status}  {symbol}  {amt_str}")
    return "\n".join(lines)


def format_guardrails(g: Mapping[str, Any]) -> str:
    cfg = g.get("guardrails") or g
    lines = ["[GUARDRAILS]"]
    for k, v in cfg.items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("Edit with: /set <key> <value>")
    return "\n".join(lines)


def format_panic_result(result: Mapping[str, Any]) -> str:
    scope = result.get("scope", "?")
    attempted = int(result.get("closed_attempted") or 0)
    ok = int(result.get("closed_ok") or 0)
    lines = [
        "[PANIC]",
        f"scope:     {scope}",
        f"attempted: {attempted}",
        f"closed:    {ok}",
        f"now paused: {result.get('now_paused', True)}",
    ]
    failures = [r for r in (result.get("results") or []) if r.get("status") != "ok"]
    if failures:
        lines.append("")
        lines.append("Failed:")
        for r in failures[:10]:
            lines.append(f"  pos#{r.get('position_id')} ({r.get('instrument_id')}): {r.get('detail')}")
    return "\n".join(lines)


def format_help() -> str:
    return (
        "Trading bot Telegram control surface.\n"
        "\n"
        "Commands:\n"
        "  /status            current bot state\n"
        "  /portfolio         positions + equity / available / P&L\n"
        "  /universe          instruments currently tracked\n"
        "  /signals           explain the live entry/exit rules + tools\n"
        "  /history [N]       last N trade-execution outcomes (default 20)\n"
        "  /guardrails        show current guardrails\n"
        "  /set <k> <v>       update a guardrails field\n"
        "  /pause             pause the trading loop (positions are kept)\n"
        "  /start             resume the trading loop\n"
        "  /stop              same as /pause\n"
        "  /resume            same as /start\n"
        "  /panic             close ALL open positions (incl. manual) and pause\n"
        "  /panic_bot_only    close only bot-owned positions and pause\n"
        "  /ask <question>    ask the LLM about the bot's state\n"
        "  /alerts            toggle which Telegram alerts you receive\n"
        "  /help              this message\n"
        "\n"
        "Anything that's not a command is treated as /ask <text>."
    )


def format_signals(payload: Mapping[str, Any]) -> str:
    """Render the structured rule set + tool catalog as a Telegram message."""
    from ..strategy.rules_summary import render_rules_text

    lines = [render_rules_text(payload)]
    perf = payload.get("tool_performance") or []
    if perf:
        lines.append("")
        lines.append("Tool performance (rolling):")
        # Show only tools with at least one observation, top 8 by hit_rate.
        ranked = sorted(
            (p for p in perf if int(p.get("observations", 0)) > 0),
            key=lambda p: float(p.get("hit_rate", 0.0)),
            reverse=True,
        )
        for p in ranked[:8]:
            lines.append(
                f"  {str(p.get('tool_name', '?'))[:18]:18s}  "
                f"hits {int(p.get('hits', 0))}/{int(p.get('observations', 0))}"
                f"  rate {float(p.get('hit_rate', 0.0)):.2f}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_unix(ts: Any) -> str:
    if ts is None:
        return "—"
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%H:%M:%S UTC")
    except (TypeError, ValueError):
        return "—"


def _money(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):,.2f}"
    except (TypeError, ValueError):
        return str(v)
