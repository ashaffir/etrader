"""Command parsing + dispatch for the Telegram service.

Splits incoming Telegram text into a (command, args) tuple, then maps
commands to control-API calls. Keeping this file separate from the
polling loop and the formatters makes each piece independently
testable.

Most handlers return plain ``str`` (the reply text). The /alerts
handler also wants to attach an inline keyboard, so handlers may
return a :class:`CommandReply` with both text and ``reply_markup``.
``dispatch()`` always returns a ``CommandReply`` for uniformity; the
bot loop unpacks both fields when calling ``send_message``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from .alerts_menu import build_alerts_caption, build_alerts_keyboard
from .channel_formatter import (
    format_channels_help,
    format_channels_logs,
    format_channels_overview,
    format_channels_test,
    parse_channels_args,
)
from .control_client import ControlAPIClient, ControlAPIError
from .directives_formatter import (
    format_clear_result as format_directive_clear,
    format_directives,
    format_note_result,
    format_set_result as format_directive_set,
)
from .formatters import (
    format_fundamentals,
    format_guardrails,
    format_help,
    format_history,
    format_news,
    format_panic_result,
    format_portfolio,
    format_signals,
    format_status,
    format_universe,
)
from .tokens_formatter import format_tokens
from .stats_formatter import (
    format_by_symbol as format_stats_by_symbol,
    format_closed as format_stats_closed,
    format_daily as format_stats_daily,
    format_open as format_stats_open,
    format_overview as format_stats_overview,
    format_window as format_stats_window,
)
from .stats_menu import build_stats_caption, build_stats_keyboard


@dataclass(frozen=True)
class ParsedCommand:
    name: str       # canonical lowercase command, no leading "/"
    args: str       # everything after the first space (raw)


def parse_command(raw_text: str) -> ParsedCommand:
    """Normalize a Telegram message into a ``ParsedCommand``.

    Telegram sometimes sends ``/cmd@MyBot`` (when in a group); we strip
    the bot suffix. Non-command text is mapped to ``ask`` so plain
    questions are forwarded to the LLM.
    """
    text = (raw_text or "").strip()
    if not text:
        return ParsedCommand(name="", args="")
    if not text.startswith("/"):
        return ParsedCommand(name="ask", args=text)
    head, _, rest = text.partition(" ")
    head = head[1:]  # strip leading '/'
    head = head.split("@", 1)[0].lower()
    return ParsedCommand(name=head, args=rest.strip())


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CommandReply:
    """Return value for a command handler.

    Most handlers only fill ``text``. The /alerts handler also fills
    ``reply_markup`` with a Telegram inline keyboard so the message
    arrives with tappable toggles.
    """

    text: str
    reply_markup: dict[str, Any] | None = None
    chat_id: int | None = None  # populated by dispatch from CommandContext


HandlerReturn = "str | CommandReply"
CommandHandler = Callable[["CommandContext"], HandlerReturn]


@dataclass
class CommandContext:
    api: ControlAPIClient
    cmd: ParsedCommand
    sender_username: str | None
    logger: logging.Logger
    chat_id: int = 0  # set by the polling layer; used for /alerts subscriptions


def dispatch(ctx: CommandContext) -> CommandReply:
    """Resolve ``ctx.cmd`` to a handler; returns the rendered Telegram reply.

    Always returns a :class:`CommandReply` so the bot loop has a
    uniform shape regardless of whether the handler attached a
    keyboard. ``ControlAPIError`` is caught here so a single-command
    failure can never crash the polling loop.
    """
    handler = _COMMANDS.get(ctx.cmd.name) or _unknown
    try:
        result = handler(ctx)
    except ControlAPIError as exc:
        return CommandReply(text=f"Control API error: {exc}")
    if isinstance(result, CommandReply):
        return result
    return CommandReply(text=str(result))


# -- handlers --------------------------------------------------------------

def _h_help(_ctx: CommandContext) -> str:
    return format_help()


def _h_status(ctx: CommandContext) -> str:
    return format_status(ctx.api.status())


def _h_portfolio(ctx: CommandContext) -> str:
    return format_portfolio(ctx.api.portfolio())


def _h_universe(ctx: CommandContext) -> str:
    return format_universe(ctx.api.universe())


def _h_news(ctx: CommandContext) -> str:
    limit = 25
    if ctx.cmd.args:
        try:
            limit = max(1, min(100, int(ctx.cmd.args.split()[0])))
        except (ValueError, IndexError):
            pass
    return format_news(ctx.api.news(limit=limit))


def _h_channels(ctx: CommandContext) -> str:
    """``/channels [test|logs] [name1,name2,...]`` — inspect news sources."""
    raw = (ctx.cmd.args or "").strip()
    if raw.lower() in {"help", "?", "-h", "--help"}:
        return format_channels_help()

    subcommand, names = parse_channels_args(raw)
    if subcommand == "test":
        payload = ctx.api.news_channels_test(only=names or None)
        return format_channels_test(payload)
    if subcommand == "logs":
        return format_channels_logs(ctx.api.news_channels())
    return format_channels_overview(ctx.api.news_channels())


def _h_fundamentals(ctx: CommandContext) -> str:
    """``/fundamentals [SYMBOL]`` — list cached symbols or one detail view."""
    raw = ctx.cmd.args.strip()
    symbol = raw.split()[0] if raw else None
    return format_fundamentals(ctx.api.fundamentals(symbol=symbol))


def _h_signals(ctx: CommandContext) -> str:
    return format_signals(ctx.api.strategy_signals())


def _h_history(ctx: CommandContext) -> str:
    limit = 20
    if ctx.cmd.args:
        try:
            limit = max(1, min(100, int(ctx.cmd.args.split()[0])))
        except (ValueError, IndexError):
            pass
    payload = ctx.api.history(limit=limit)
    return format_history(payload.get("entries") or [])


def _h_guardrails(ctx: CommandContext) -> str:
    return format_guardrails(ctx.api.get_guardrails())


def _h_set(ctx: CommandContext) -> str:
    parts = ctx.cmd.args.split(maxsplit=1)
    if len(parts) != 2:
        return "Usage: /set <key> <value>\nExample: /set max_per_trade_usd 250"
    key, value = parts[0].strip(), parts[1].strip()
    result = ctx.api.set_guardrail(key, value)
    return (
        f"Updated guardrails.{result.get('key')}\n"
        f"  before: {result.get('previous')}\n"
        f"  after:  {result.get('current')}"
    )


def _h_pause(ctx: CommandContext) -> str:
    result = ctx.api.pause(reason=f"telegram:{ctx.sender_username or '?'}")
    if result.get("was_already_paused"):
        return "Bot was already paused."
    return "Bot paused. Cycles will skip until /start (or /resume)."


def _h_resume(ctx: CommandContext) -> str:
    result = ctx.api.resume()
    if result.get("was_already_running"):
        return "Bot was already running."
    return "Bot resumed. Next cycle will run in <= check_interval_seconds."


def _h_unhalt(ctx: CommandContext) -> str:
    result = ctx.api.unhalt(reason=f"telegram:{ctx.sender_username or '?'}")
    if not result.get("was_halted"):
        return "Bot was not halted — nothing to clear."
    return (
        "Daily-loss kill switch cleared. Equity baseline will rebase on the "
        "next cycle. Set `daily_loss_stop_usd` to 0 via /set if you want to "
        "disable the kill switch permanently."
    )


def _h_panic(ctx: CommandContext) -> str:
    result = ctx.api.panic(scope="all", reason=f"telegram:{ctx.sender_username or '?'}")
    return format_panic_result(result)


def _h_panic_bot_only(ctx: CommandContext) -> str:
    result = ctx.api.panic(scope="bot_owned", reason=f"telegram:{ctx.sender_username or '?'}")
    return format_panic_result(result)


def _h_ask(ctx: CommandContext) -> str:
    question = ctx.cmd.args.strip()
    if not question:
        return "Usage: /ask <free-text question about the bot>"
    result = ctx.api.ask(question)
    answer = result.get("answer") or "(no answer)"
    latency = result.get("latency_ms")
    suffix = f"\n\n(LLM {latency} ms)" if isinstance(latency, int) else ""
    return f"{answer}{suffix}"


def _h_alerts(ctx: CommandContext) -> CommandReply:
    """Render the /alerts inline keyboard with current subscription state."""
    if ctx.chat_id <= 0:
        return CommandReply(text="Cannot determine chat_id for /alerts.")
    payload = ctx.api.alert_subscriptions(ctx.chat_id)
    return CommandReply(
        text=build_alerts_caption(payload),
        reply_markup=build_alerts_keyboard(payload),
    )


def _h_directives(ctx: CommandContext) -> str:
    return format_directives(ctx.api.directives())


def _h_directive(ctx: CommandContext) -> str:
    """``/directive set|clear <key> [value]`` — edit one directive."""
    parts = ctx.cmd.args.split(maxsplit=2)
    if not parts:
        return _directive_usage()
    sub = parts[0].lower()
    if sub == "set":
        if len(parts) < 3:
            return _directive_usage()
        key, raw_value = parts[1].strip(), parts[2].strip()
        result = ctx.api.set_directive(key, raw_value)
        return format_directive_set(result)
    if sub == "clear":
        if len(parts) < 2:
            return _directive_usage()
        key = parts[1].strip()
        result = ctx.api.clear_directive(key)
        return format_directive_clear(result)
    return _directive_usage()


def _directive_usage() -> str:
    return (
        "Usage:\n"
        "  /directive set <key> <value>\n"
        "  /directive clear <key>\n\n"
        "Known keys: no_overnight, hold_ceiling_minutes, blocked_symbols,\n"
        "            blocked_sectors, max_total_account_invested_usd\n\n"
        "Examples:\n"
        "  /directive set no_overnight true\n"
        "  /directive set blocked_symbols NVDA,TSLA\n"
        "  /directive set hold_ceiling_minutes 120\n"
        "  /directive clear blocked_symbols"
    )


def _h_note(ctx: CommandContext) -> str:
    """``/note add|set|clear [text]`` — edit the free-text directive."""
    raw = ctx.cmd.args.strip()
    if not raw:
        return _note_usage()
    head, _, rest = raw.partition(" ")
    sub = head.lower()
    if sub == "clear":
        result = ctx.api.clear_directive_note()
        return format_note_result(result, cleared=True)
    if sub in {"add", "set"}:
        text = rest.strip()
        if not text:
            return _note_usage()
        result = ctx.api.set_directive_note(text)
        return format_note_result(result, cleared=False)
    # No sub-command given — treat the entire arg as the new note text.
    result = ctx.api.set_directive_note(raw)
    return format_note_result(result, cleared=False)


def _note_usage() -> str:
    return (
        "Usage:\n"
        "  /note add <text>     — replace the current note\n"
        "  /note set <text>     — alias of add\n"
        "  /note clear          — remove the current note\n\n"
        "Notes are surfaced to the manager LLM in every cycle prompt; "
        "use them for soft preferences the structured directive schema "
        "doesn't capture."
    )


def _h_tokens(ctx: CommandContext) -> str:
    return format_tokens(ctx.api.tokens())


def _h_stats(ctx: CommandContext) -> CommandReply:
    """``/stats [view]`` — performance dashboard.

    Without args, presents an inline keyboard menu with the views
    (Overview / Today / 7d / 30d / All-time / Open / Closed / By
    Symbol / Daily). With an arg, jumps straight to that view (so
    power users can type ``/stats today`` etc.). Valid args match
    :data:`stats_menu.STATS_VIEWS` keys plus their natural aliases.
    """
    arg = (ctx.cmd.args or "").strip().lower()
    if not arg:
        return CommandReply(
            text=build_stats_caption(),
            reply_markup=build_stats_keyboard(),
        )
    view = _resolve_stats_view_alias(arg)
    if view is None:
        return CommandReply(
            text=(
                "Unknown /stats view. Valid: overview, today, 7d, 30d, all, "
                "open, closed, by-symbol, daily."
            ),
            reply_markup=build_stats_keyboard(),
        )
    return CommandReply(text=render_stats_view(ctx.api, view))


_STATS_VIEW_ALIASES: dict[str, str] = {
    "overview": "overview", "summary": "overview",
    "today": "today", "now": "today",
    "7d": "7d", "week": "7d", "7days": "7d",
    "30d": "30d", "month": "30d", "30days": "30d",
    "all": "all", "all-time": "all", "alltime": "all", "lifetime": "all",
    "open": "open", "open-positions": "open", "positions": "open",
    "closed": "closed", "trades": "closed", "history": "closed",
    "by-symbol": "by_symbol", "by_symbol": "by_symbol", "symbols": "by_symbol", "symbol": "by_symbol",
    "daily": "daily", "days": "daily",
}


def _resolve_stats_view_alias(arg: str) -> str | None:
    return _STATS_VIEW_ALIASES.get(arg.replace(" ", ""))


def render_stats_view(api: ControlAPIClient, view: str) -> str:
    """Fetch + format one stats view. Exported so the callback layer can reuse."""
    if view == "overview":
        return format_stats_overview(api.stats_summary())
    if view in ("today", "7d", "30d", "all"):
        return format_stats_window(api.stats_summary(), period=view)
    if view == "open":
        return format_stats_open(api.stats_summary())
    if view == "closed":
        return format_stats_closed(api.stats_closed(limit=50))
    if view == "by_symbol":
        return format_stats_by_symbol(api.stats_by_symbol())
    if view == "daily":
        return format_stats_daily(api.stats_daily(limit=30))
    return f"Unknown view: {view}"


def _unknown(ctx: CommandContext) -> str:
    return (
        f"Unknown command /{ctx.cmd.name}.\n\n"
        + format_help()
    )


_COMMANDS: dict[str, CommandHandler] = {
    "help": _h_help,
    "status": _h_status,
    "portfolio": _h_portfolio,
    "universe": _h_universe,
    "news": _h_news,
    "channels": _h_channels,
    "sources": _h_channels,  # alias
    "feeds": _h_channels,    # alias
    "fundamentals": _h_fundamentals,
    "fund": _h_fundamentals,  # alias
    "signals": _h_signals,
    "rules": _h_signals,        # alias
    "strategy": _h_signals,     # alias
    "history": _h_history,
    "guardrails": _h_guardrails,
    "config": _h_guardrails,  # alias
    "set": _h_set,
    "pause": _h_pause,
    "stop": _h_pause,         # alias — user-facing /stop pauses the loop
    "resume": _h_resume,
    "start": _h_resume,       # alias — /start resumes
    "unhalt": _h_unhalt,
    "panic": _h_panic,
    "panic_bot_only": _h_panic_bot_only,
    "panicbotonly": _h_panic_bot_only,  # tolerant
    "ask": _h_ask,
    "alerts": _h_alerts,
    "subscriptions": _h_alerts,  # alias
    "stats": _h_stats,
    "performance": _h_stats,     # alias
    "perf": _h_stats,            # alias
    "directives": _h_directives,
    "directive": _h_directive,
    "rules_persistent": _h_directives,  # readable alias
    "note": _h_note,
    "notes": _h_note,           # alias
    "tokens": _h_tokens,
    "cost": _h_tokens,          # alias
    "usage": _h_tokens,         # alias
}
