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
    source_counts = universe.get("source_counts") or {}
    reasons = universe.get("reasons") or {}
    rejected = universe.get("rejected") or {}
    if not symbols:
        body = "[UNIVERSE]\n(no instruments tracked — the news scan returned no candidates that passed the activity filter)"
        if rejected:
            body += "\n\nRecent rejections:\n"
            body += "\n".join(f"  {sym}: {reason}" for sym, reason in list(rejected.items())[:10])
        return body
    src_summary = ", ".join(f"{k}={v}" for k, v in source_counts.items()) or "(no breakdown)"
    lines = [
        "[UNIVERSE]",
        f"tracking {len(symbols)} instrument(s) — {src_summary}",
        "",
    ]
    for sym in symbols:
        reason = reasons.get(sym) or "(no reason recorded)"
        lines.append(f"  {sym}: {reason}")
    if rejected:
        lines.append("")
        lines.append(f"Rejected this refresh ({len(rejected)}):")
        for sym, reason in list(rejected.items())[:10]:
            lines.append(f"  {sym}: {reason}")
        if len(rejected) > 10:
            lines.append(f"  … and {len(rejected) - 10} more")
    return "\n".join(lines)


def format_news(news: Mapping[str, Any]) -> str:
    """Render the /news payload returned by ``BotController.snapshot_news``."""
    candidates = list(news.get("candidates") or [])
    last_scan = news.get("last_scan")
    next_scan_in_s = news.get("next_scan_in_seconds")

    lines = ["[NEWS]"]
    if last_scan:
        finished = _fmt_unix(last_scan.get("finished_at_unix"))
        per_src = last_scan.get("per_source_counts") or {}
        src_summary = ", ".join(f"{k}={v}" for k, v in per_src.items()) or "(no sources)"
        lines.append(
            f"last scan: {finished} — kept {last_scan.get('items_kept', 0)}, "
            f"obs {last_scan.get('observations_recorded', 0)}, {src_summary}"
        )
        errs = last_scan.get("per_source_errors") or {}
        if errs:
            err_str = ", ".join(f"{k}: {v[:30]}" for k, v in errs.items())
            lines.append(f"errors: {err_str}")
    else:
        lines.append("last scan: (none yet)")
    if next_scan_in_s is not None:
        mins = max(0, int(float(next_scan_in_s) / 60))
        lines.append(f"next scan in: ~{mins} min")
    lines.append("")
    if not candidates:
        lines.append("(candidate store empty)")
        return "\n".join(lines)
    lines.append(f"top {len(candidates)} candidates:")
    for c in candidates:
        sym = c.get("symbol", "?")
        score = float(c.get("score") or 0.0)
        sources = "+".join(c.get("sources") or [])
        head = (c.get("headlines") or [None])[0] or ""
        lines.append(f"  {sym:<8}  score={score:5.2f}  [{sources}]  {head[:70]}")
    return "\n".join(lines)


def format_fundamentals(payload: Mapping[str, Any]) -> str:
    """Render the /fundamentals control-API payload.

    Two shapes:

    - ``payload["symbol"]`` set → single-symbol detail view. Falls back
      to a "not cached" message when ``snapshot is None``.
    - No symbol → summary list of every cached entry, grouped by sector.
    """
    if not payload.get("enabled", True):
        return (
            "[FUNDAMENTALS]\n"
            "Fundamentals cache is disabled in config "
            "(set `[fundamentals] enabled = true` and restart)."
        )
    if payload.get("symbol"):
        return _format_fundamentals_detail(payload)
    return _format_fundamentals_list(payload)


def _format_fundamentals_list(payload: Mapping[str, Any]) -> str:
    items = list(payload.get("items") or [])
    if not items:
        return "[FUNDAMENTALS]\n(cache empty — wait for the next universe refresh)"
    lines = [f"[FUNDAMENTALS]  ({payload.get('count', len(items))} cached)"]
    by_sector: dict[str, list[Mapping[str, Any]]] = {}
    for it in items:
        sector = it.get("sector") or "(no sector)"
        by_sector.setdefault(sector, []).append(it)
    for sector in sorted(by_sector.keys()):
        lines.append("")
        lines.append(f"{sector}:")
        for it in sorted(by_sector[sector], key=lambda i: i.get("symbol") or ""):
            sym = (it.get("symbol") or "?").ljust(8)
            name = (it.get("name") or "")[:30]
            age_h = _hours_since(it.get("fetched_at_unix"))
            lines.append(f"  {sym}  {name:<30}  age={age_h}")
    lines.append("")
    lines.append("Use `/fundamentals <SYMBOL>` for full detail.")
    return "\n".join(lines)


def _format_fundamentals_detail(payload: Mapping[str, Any]) -> str:
    sym = payload.get("symbol") or "?"
    snap = payload.get("snapshot")
    if not snap:
        return (
            f"[FUNDAMENTALS] {sym}\n"
            "(not cached — try again after the next universe refresh)"
        )
    name = snap.get("name") or sym
    sector = snap.get("sector") or "(no sector)"
    industry = snap.get("industry") or "(no industry)"
    quote_type = snap.get("quote_type") or "?"
    currency = snap.get("currency") or ""
    age_h = _hours_since(snap.get("fetched_at_unix"))
    lines = [
        f"[FUNDAMENTALS] {sym}  —  {name}",
        f"type={quote_type}  sector={sector} / {industry}  currency={currency}  age={age_h}",
        "",
        "Valuation:",
        f"  market_cap:        {_compact_money(snap.get('market_cap'))}",
        f"  enterprise_value:  {_compact_money(snap.get('enterprise_value'))}",
        f"  P/E (trailing):    {_fmt_ratio(snap.get('trailing_pe'))}",
        f"  P/E (forward):     {_fmt_ratio(snap.get('forward_pe'))}",
        f"  P/B:               {_fmt_ratio(snap.get('price_to_book'))}",
        f"  P/S:               {_fmt_ratio(snap.get('price_to_sales'))}",
        f"  dividend yield:    {_fmt_pct(snap.get('dividend_yield'))}",
        f"  beta:              {_fmt_ratio(snap.get('beta'))}",
        "",
        "Profitability / growth:",
        f"  profit margin:     {_fmt_pct(snap.get('profit_margin'))}",
        f"  operating margin:  {_fmt_pct(snap.get('operating_margin'))}",
        f"  ROE:               {_fmt_pct(snap.get('return_on_equity'))}",
        f"  revenue growth:    {_fmt_pct(snap.get('revenue_growth'))}",
        f"  earnings growth:   {_fmt_pct(snap.get('earnings_growth'))}",
        f"  debt/equity:       {_fmt_ratio(snap.get('debt_to_equity'))}",
        "",
        "52-week range:",
        f"  high:              {_compact_money(snap.get('fifty_two_week_high'))}",
        f"  low:               {_compact_money(snap.get('fifty_two_week_low'))}",
    ]
    target = snap.get("analyst_target_mean")
    rec = snap.get("analyst_recommendation")
    count = snap.get("analyst_count")
    if target or rec or count:
        lines.append("")
        lines.append("Analyst consensus:")
        lines.append(f"  recommendation:    {rec or '—'}")
        lines.append(f"  target (mean):     {_compact_money(target)}")
        lines.append(f"  # analysts:        {count if count is not None else '—'}")
    next_e = snap.get("next_earnings_unix")
    if next_e:
        lines.append("")
        lines.append(f"Next earnings: {_fmt_unix(next_e)}")
    summary_text = snap.get("summary")
    if summary_text:
        lines.append("")
        lines.append(str(summary_text))
    return "\n".join(lines)


def _hours_since(ts: Any) -> str:
    if ts is None:
        return "—"
    try:
        delta = max(0.0, datetime.now(timezone.utc).timestamp() - float(ts))
    except (TypeError, ValueError):
        return "—"
    if delta < 3600:
        return f"{int(delta / 60)}m"
    return f"{delta / 3600:.1f}h"


def _compact_money(v: Any) -> str:
    if v is None:
        return "—"
    try:
        n = float(v)
    except (TypeError, ValueError):
        return str(v)
    sign = "-" if n < 0 else ""
    a = abs(n)
    if a >= 1e12:
        return f"{sign}{a / 1e12:.2f}T"
    if a >= 1e9:
        return f"{sign}{a / 1e9:.2f}B"
    if a >= 1e6:
        return f"{sign}{a / 1e6:.2f}M"
    if a >= 1e3:
        return f"{sign}{a / 1e3:.2f}K"
    return f"{sign}{a:.2f}"


def _fmt_ratio(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_pct(v: Any) -> str:
    if v is None:
        return "—"
    try:
        # yfinance returns ratios as fractions for margins/growth (0.27 = 27%),
        # but dividend_yield is already a percentage in newer versions. We
        # render the fractional shape with a small heuristic: ratios < 5
        # are treated as fractions, larger values as already-percentages.
        n = float(v)
    except (TypeError, ValueError):
        return str(v)
    if abs(n) < 5:
        n *= 100.0
    return f"{n:+.2f}%"


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
        "  /universe          instruments currently tracked (with reasons)\n"
        "  /news [N]          top-N news candidates + last scan stats\n"
        "  /channels [sub]    news-source health (sub: test [names] | logs)\n"
        "  /fundamentals [SYM] cached fundamentals (list or one symbol)\n"
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
        "  /stats [view]      performance dashboard (menu, or e.g. /stats today)\n"
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
