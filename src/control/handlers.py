"""HTTP route handlers — thin glue between BotController and JSON.

Each handler is a pure function ``(controller, body, query) -> (status, payload)``.
We register routes in :func:`build_route_table` so the HTTP layer can
look them up by ``(method, path)`` without an import cycle.

The handlers never touch eToro / Azure directly: that's the controller's
job. We just translate request shape ↔ response shape.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from .controller import BotController, ControllerError


HandlerFn = Callable[[BotController, Mapping[str, Any] | None, Mapping[str, str]], tuple[int, Any]]


class RouteTable:
    """Tiny ``(method, path) -> handler`` map."""

    def __init__(self) -> None:
        self._map: dict[tuple[str, str], HandlerFn] = {}

    def add(self, method: str, path: str, handler: HandlerFn) -> None:
        self._map[(method.upper(), path)] = handler

    def lookup(self, method: str, path: str) -> HandlerFn:
        try:
            return self._map[(method.upper(), path)]
        except KeyError as exc:
            raise KeyError(f"no handler for {method} {path}") from exc

    def routes(self) -> list[tuple[str, str]]:
        return sorted(self._map.keys())


# ---------------------------------------------------------------------------
# Individual handlers
# ---------------------------------------------------------------------------

def _ping(_c: BotController, _body, _q) -> tuple[int, Any]:
    return 200, {"ok": True}


def _status(c: BotController, _body, _q) -> tuple[int, Any]:
    return 200, c.snapshot_status_dict()


def _portfolio(c: BotController, _body, _q) -> tuple[int, Any]:
    return 200, c.snapshot_portfolio()


def _universe(c: BotController, _body, _q) -> tuple[int, Any]:
    return 200, c.snapshot_universe()


def _news(c: BotController, _body, q: Mapping[str, str]) -> tuple[int, Any]:
    raw = q.get("limit") or "25"
    try:
        limit = int(raw)
    except ValueError:
        limit = 25
    return 200, c.snapshot_news(limit=limit)


def _fundamentals(
    c: BotController, _body, q: Mapping[str, str],
) -> tuple[int, Any]:
    """``GET /fundamentals[?symbol=AAPL]`` — list cached symbols or one detail."""
    symbol = (q.get("symbol") or "").strip() or None
    return 200, c.snapshot_fundamentals(symbol=symbol)


def _history(c: BotController, _body, q: Mapping[str, str]) -> tuple[int, Any]:
    raw = q.get("limit") or "20"
    try:
        limit = int(raw)
    except ValueError:
        limit = 20
    return 200, {"entries": c.recent_history(limit=limit)}


def _pause(c: BotController, body: Mapping[str, Any] | None, _q) -> tuple[int, Any]:
    reason = (body or {}).get("reason")
    return 200, c.pause(reason=reason)


def _resume(c: BotController, _body, _q) -> tuple[int, Any]:
    return 200, c.resume()


def _unhalt(c: BotController, body: Mapping[str, Any] | None, _q) -> tuple[int, Any]:
    reason = (body or {}).get("reason")
    return 200, c.unhalt(reason=reason)


def _panic(c: BotController, body: Mapping[str, Any] | None, _q) -> tuple[int, Any]:
    body = body or {}
    scope = str(body.get("scope") or "all").lower()
    reason = body.get("reason")
    if scope not in {"all", "bot_owned"}:
        raise ControllerError("scope must be 'all' or 'bot_owned'")
    return 200, c.panic_close_all(scope=scope, reason=reason)


def _config_get(c: BotController, _body, _q) -> tuple[int, Any]:
    return 200, {"guardrails": c.get_guardrails()}


def _config_set(c: BotController, body: Mapping[str, Any] | None, _q) -> tuple[int, Any]:
    body = body or {}
    key = body.get("key")
    if not key or not isinstance(key, str):
        raise ControllerError("body must include 'key' (string)")
    if "value" not in body:
        raise ControllerError("body must include 'value'")
    return 200, c.apply_guardrails_change(key, body.get("value"))


def _ask(c: BotController, body: Mapping[str, Any] | None, _q) -> tuple[int, Any]:
    body = body or {}
    question = body.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ControllerError("body must include 'question' (string)")
    return 200, c.ask(question)


def _strategy_signals(c: BotController, _body, _q) -> tuple[int, Any]:
    return 200, c.snapshot_strategy_rules()


def _stats_summary(c: BotController, _body, _q) -> tuple[int, Any]:
    """``GET /stats`` — overview payload (multi-window + open positions)."""
    return 200, c.snapshot_performance()


def _stats_by_symbol(c: BotController, _body, _q) -> tuple[int, Any]:
    """``GET /stats/by-symbol`` — per-symbol rollup."""
    return 200, c.snapshot_performance_symbols()


def _stats_closed(c: BotController, _body, q: Mapping[str, str]) -> tuple[int, Any]:
    """``GET /stats/closed?limit=N`` — most-recent closed trades."""
    try:
        limit = int(q.get("limit") or "50")
    except ValueError:
        limit = 50
    return 200, c.snapshot_performance_closed_trades(limit=limit)


def _stats_daily(c: BotController, _body, q: Mapping[str, str]) -> tuple[int, Any]:
    """``GET /stats/daily?limit=N`` — daily snapshots."""
    try:
        limit = int(q.get("limit") or "30")
    except ValueError:
        limit = 30
    return 200, c.snapshot_performance_dailies(limit=limit)


# ---------------------------------------------------------------------------
# Operator directives
# ---------------------------------------------------------------------------

def _directives_get(c: BotController, _body, _q) -> tuple[int, Any]:
    return 200, c.snapshot_directives()


def _directives_set(
    c: BotController, body: Mapping[str, Any] | None, _q,
) -> tuple[int, Any]:
    body = body or {}
    key = body.get("key")
    if not isinstance(key, str) or not key.strip():
        raise ControllerError("body must include 'key' (string)")
    if "value" not in body:
        raise ControllerError("body must include 'value'")
    return 200, c.set_directive(key.strip(), body.get("value"))


def _directives_clear(
    c: BotController, body: Mapping[str, Any] | None, _q,
) -> tuple[int, Any]:
    body = body or {}
    key = body.get("key")
    if not isinstance(key, str) or not key.strip():
        raise ControllerError("body must include 'key' (string)")
    return 200, c.clear_directive(key.strip())


def _directives_note_set(
    c: BotController, body: Mapping[str, Any] | None, _q,
) -> tuple[int, Any]:
    body = body or {}
    text = body.get("text")
    if text is None:
        raise ControllerError("body must include 'text' (string)")
    return 200, c.set_directive_note(str(text))


def _directives_note_clear(c: BotController, _body, _q) -> tuple[int, Any]:
    return 200, c.clear_directive_note()


# ---------------------------------------------------------------------------
# LLM token usage
# ---------------------------------------------------------------------------

def _tokens_get(c: BotController, _body, _q) -> tuple[int, Any]:
    return 200, c.snapshot_token_usage()


def _news_channels(c: BotController, _body, _q) -> tuple[int, Any]:
    """``GET /news/channels`` — per-source health overview."""
    return 200, c.snapshot_news_channels()


def _news_channels_test(
    c: BotController, body: Mapping[str, Any] | None, q: Mapping[str, str],
) -> tuple[int, Any]:
    """``POST /news/channels/test`` — live dry-run against each source.

    Optional body / query parameter ``only`` accepts a comma-separated
    list (query) or list/string (body) of source names to probe. When
    omitted, every wired source is probed.
    """
    only: list[str] | None = None
    raw = (body or {}).get("only") if body else None
    if raw is None:
        raw = q.get("only")
    if isinstance(raw, str):
        only = [s for s in (p.strip() for p in raw.split(",")) if s]
    elif isinstance(raw, (list, tuple)):
        only = [str(s).strip() for s in raw if str(s).strip()]
    return 200, c.test_news_channels(only=only)


# ---------------------------------------------------------------------------
# Alerts: list types, list/edit per-chat subscriptions, drain pending queue
# ---------------------------------------------------------------------------

def _alerts_types(c: BotController, _body, _q) -> tuple[int, Any]:
    return 200, {"types": c.list_alert_types()}


def _alerts_subscriptions_get(
    c: BotController, _body, q: Mapping[str, str],
) -> tuple[int, Any]:
    chat_id = _require_chat_id(q)
    return 200, c.get_alert_subscriptions(chat_id)


def _alerts_subscriptions_set(
    c: BotController, body: Mapping[str, Any] | None, q: Mapping[str, str],
) -> tuple[int, Any]:
    body = body or {}
    chat_id = _require_chat_id(q, body)
    type_str = body.get("type")
    if not isinstance(type_str, str):
        raise ControllerError("body must include 'type' (string)")
    if "toggle" in body and bool(body.get("toggle")):
        return 200, c.toggle_alert_subscription(chat_id, type_str)
    if "enabled" not in body:
        raise ControllerError("body must include 'enabled' (bool) or 'toggle': true")
    return 200, c.set_alert_subscription(chat_id, type_str, bool(body.get("enabled")))


def _alerts_pending(
    c: BotController, _body, q: Mapping[str, str],
) -> tuple[int, Any]:
    chat_id = _require_chat_id(q)
    raw_limit = q.get("limit") or "50"
    try:
        limit = int(raw_limit)
    except ValueError:
        limit = 50
    return 200, {"chat_id": chat_id, "alerts": c.drain_alerts(chat_id, limit=limit)}


def _require_chat_id(*sources: Mapping[str, Any]) -> int:
    """Return the chat_id from any of the provided dict-likes; raise if missing."""
    for src in sources:
        if not src:
            continue
        raw = src.get("chat_id")
        if raw is None:
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    raise ControllerError("chat_id required (query string ?chat_id=... or body field)")


def _shutdown(c: BotController, _body, _q) -> tuple[int, Any]:
    """Operator hard-stop: requests the trading process to exit cleanly.

    Distinct from /pause: this terminates the bot. Useful when you want
    to pull a maintenance window from Telegram before redeploying.
    """
    c.request_stop()
    return 200, {"shutdown_requested": True}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def build_route_table() -> RouteTable:
    rt = RouteTable()
    rt.add("GET", "/ping", _ping)
    rt.add("GET", "/status", _status)
    rt.add("GET", "/portfolio", _portfolio)
    rt.add("GET", "/universe", _universe)
    rt.add("GET", "/news", _news)
    rt.add("GET", "/fundamentals", _fundamentals)
    rt.add("GET", "/history", _history)
    rt.add("GET", "/config/guardrails", _config_get)
    rt.add("POST", "/config/guardrails", _config_set)
    rt.add("POST", "/pause", _pause)
    rt.add("POST", "/resume", _resume)
    rt.add("POST", "/unhalt", _unhalt)
    rt.add("POST", "/panic", _panic)
    rt.add("POST", "/ask", _ask)
    rt.add("POST", "/shutdown", _shutdown)
    rt.add("GET", "/strategy/signals", _strategy_signals)
    rt.add("GET", "/stats", _stats_summary)
    rt.add("GET", "/stats/by-symbol", _stats_by_symbol)
    rt.add("GET", "/stats/closed", _stats_closed)
    rt.add("GET", "/stats/daily", _stats_daily)
    rt.add("GET", "/news/channels", _news_channels)
    rt.add("POST", "/news/channels/test", _news_channels_test)
    rt.add("GET", "/alerts/types", _alerts_types)
    rt.add("GET", "/alerts/subscriptions", _alerts_subscriptions_get)
    rt.add("POST", "/alerts/subscriptions", _alerts_subscriptions_set)
    rt.add("GET", "/alerts/pending", _alerts_pending)
    rt.add("GET", "/directives", _directives_get)
    rt.add("POST", "/directives", _directives_set)
    rt.add("POST", "/directives/clear", _directives_clear)
    rt.add("POST", "/directives/note", _directives_note_set)
    rt.add("POST", "/directives/note/clear", _directives_note_clear)
    rt.add("GET", "/tokens", _tokens_get)
    return rt
