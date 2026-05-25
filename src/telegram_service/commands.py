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
    "panic": _h_panic,
    "panic_bot_only": _h_panic_bot_only,
    "panicbotonly": _h_panic_bot_only,  # tolerant
    "ask": _h_ask,
    "alerts": _h_alerts,
    "subscriptions": _h_alerts,  # alias
}
